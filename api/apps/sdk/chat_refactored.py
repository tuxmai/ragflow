"""
Chat Assistant Management API Module

This module handles chat assistant (dialog) management operations including creation,
updating, deletion, and listing of chat assistants. It provides comprehensive
validation, error handling, and logging for all chat-related operations.

Key Features:
- Chat assistant lifecycle management (CRUD operations)
- Dataset integration and validation
- LLM model configuration and validation
- Prompt configuration management
- Comprehensive error handling and logging
- Tenant-based access control

Author: RAGFlow Team
License: Apache License 2.0
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from flask import request

from api import settings
from api.db import StatusEnum
from api.db.services.dialog_service import DialogService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.llm_service import TenantLLMService
from api.db.services.user_service import TenantService
from api.utils import get_uuid
from api.utils.api_utils import check_duplicate_ids, get_error_data_result, get_result, token_required

# Configure module-specific logger
logger = logging.getLogger(__name__)


class ChatAssistantError(Exception):
    """Custom exception for chat assistant-related errors."""

    def __init__(self, message: str, code: int = 500):
        self.message = message
        self.code = code
        super().__init__(self.message)


class DatasetValidator:
    """Validates dataset access and configuration for chat assistants."""

    @staticmethod
    def validate_dataset_access(dataset_ids: List[str], tenant_id: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that the tenant has access to all specified datasets.

        Args:
            dataset_ids: List of dataset identifiers
            tenant_id: Tenant identifier

        Returns:
            Tuple of (is_valid, error_message)
        """
        logger.info(f"Validating dataset access for tenant {tenant_id}, datasets: {dataset_ids}")

        for dataset_id in dataset_ids:
            # Check dataset accessibility
            if not KnowledgebaseService.accessible(kb_id=dataset_id, user_id=tenant_id):
                error_msg = f"You don't own the dataset {dataset_id}"
                logger.warning(f"Dataset access denied: {error_msg}")
                return False, error_msg

            # Check dataset has parsed content
            knowledge_bases = KnowledgebaseService.query(id=dataset_id)
            if knowledge_bases and knowledge_bases[0].chunk_num == 0:
                error_msg = f"The dataset {dataset_id} doesn't have parsed files"
                logger.warning(f"Dataset validation failed: {error_msg}")
                return False, error_msg

        logger.info(f"Dataset validation successful for {len(dataset_ids)} datasets")
        return True, None

    @staticmethod
    def validate_embedding_model_consistency(dataset_ids: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate that all datasets use the same embedding model.

        Args:
            dataset_ids: List of dataset identifiers

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not dataset_ids:
            return True, None

        logger.info(f"Validating embedding model consistency for {len(dataset_ids)} datasets")

        knowledge_bases = KnowledgebaseService.get_by_ids(dataset_ids)
        embedding_model_ids = [TenantLLMService.split_model_name_and_factory(kb.embd_id)[0] for kb in knowledge_bases]

        unique_embedding_models = list(set(embedding_model_ids))

        if len(unique_embedding_models) > 1:
            error_msg = "Datasets use different embedding models"
            logger.warning(f"Embedding model consistency check failed: {error_msg}")
            return False, error_msg

        logger.info("Embedding model consistency validation successful")
        return True, None


class LLMConfigurationManager:
    """Manages LLM configuration and validation for chat assistants."""

    @staticmethod
    def process_llm_configuration(llm_config: Dict[str, Any], tenant_id: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Process and validate LLM configuration.

        Args:
            llm_config: LLM configuration data
            tenant_id: Tenant identifier

        Returns:
            Tuple of (is_valid, error_message, processed_config)
        """
        logger.info(f"Processing LLM configuration for tenant {tenant_id}")

        processed_config = {}

        if "model_name" in llm_config:
            model_name = llm_config.pop("model_name")
            processed_config["llm_id"] = model_name

            if model_name:
                # Validate model exists for tenant
                llm_name, llm_factory = TenantLLMService.split_model_name_and_factory(model_name)
                if not TenantLLMService.query(tenant_id=tenant_id, llm_name=llm_name, llm_factory=llm_factory, model_type="chat"):
                    error_msg = f"Model `{model_name}` doesn't exist"
                    logger.warning(f"LLM validation failed: {error_msg}")
                    return False, error_msg, {}

        processed_config["llm_setting"] = llm_config

        logger.info("LLM configuration processing successful")
        return True, None, processed_config

    @staticmethod
    def validate_rerank_model(rerank_model_id: str, tenant_id: str) -> Tuple[bool, Optional[str]]:
        """
        Validate rerank model availability.

        Args:
            rerank_model_id: Rerank model identifier
            tenant_id: Tenant identifier

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not rerank_model_id:
            return True, None

        logger.info(f"Validating rerank model {rerank_model_id} for tenant {tenant_id}")

        # Check built-in models
        builtin_rerank_models = ["BAAI/bge-reranker-v2-m3", "maidalun1020/bce-reranker-base_v1"]
        if rerank_model_id in builtin_rerank_models:
            return True, None

        # Check tenant-specific models
        if not TenantLLMService.query(tenant_id=tenant_id, llm_name=rerank_model_id, model_type="rerank"):
            error_msg = f"Rerank model `{rerank_model_id}` doesn't exist"
            logger.warning(f"Rerank model validation failed: {error_msg}")
            return False, error_msg

        logger.info("Rerank model validation successful")
        return True, None


class PromptConfigurationManager:
    """Manages prompt configuration and validation for chat assistants."""

    @staticmethod
    def get_default_prompt_configuration() -> Dict[str, Any]:
        """
        Get default prompt configuration.

        Returns:
            Default prompt configuration dictionary
        """
        return {
            "system": """You are an intelligent assistant. Please summarize the content of the knowledge base to answer the question. Please list the data in the knowledge base and answer in detail. When all knowledge base content is irrelevant to the question, your answer must include the sentence "The answer you are looking for is not found in the knowledge base!" Answers need to consider chat history.
      Here is the knowledge base:
      {knowledge}
      The above is the knowledge base.""",
            "prologue": "Hi! I'm your assistant, what can I do for you?",
            "parameters": [{"key": "knowledge", "optional": False}],
            "empty_response": "Sorry! No relevant content was found in the knowledge base!",
            "quote": True,
            "tts": False,
            "refine_multiturn": True,
        }

    @staticmethod
    def process_prompt_configuration(prompt_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process and merge prompt configuration with defaults.

        Args:
            prompt_config: Custom prompt configuration

        Returns:
            Processed prompt configuration
        """
        logger.info("Processing prompt configuration")

        default_config = PromptConfigurationManager.get_default_prompt_configuration()

        if not prompt_config:
            return default_config

        # Apply key mapping for backward compatibility
        key_mapping = {
            "parameters": "variables",
            "prologue": "opener",
            "quote": "show_quote",
            "system": "prompt",
            "rerank_id": "rerank_model",
            "vector_similarity_weight": "keywords_similarity_weight",
        }

        # Rename keys according to mapping
        for new_key, old_key in key_mapping.items():
            if old_key in prompt_config:
                prompt_config[new_key] = prompt_config.pop(old_key)

        # Merge with defaults
        for key, default_value in default_config.items():
            if key not in prompt_config or (not prompt_config[key] and key == "system"):
                prompt_config[key] = default_value

        logger.info("Prompt configuration processing completed")
        return prompt_config

    @staticmethod
    def validate_prompt_parameters(prompt_config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate that all required parameters are used in the system prompt.

        Args:
            prompt_config: Prompt configuration to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        logger.info("Validating prompt parameters")

        system_prompt = prompt_config.get("system", "")
        parameters = prompt_config.get("parameters", [])

        for parameter in parameters:
            if parameter.get("optional"):
                continue

            parameter_key = parameter.get("key")
            if not parameter_key:
                continue

            parameter_placeholder = f"{{{parameter_key}}}"
            if parameter_placeholder not in system_prompt:
                error_msg = f"Parameter '{parameter_key}' is not used in the system prompt"
                logger.warning(f"Parameter validation failed: {error_msg}")
                return False, error_msg

        logger.info("Prompt parameter validation successful")
        return True, None


class ChatAssistantManager:
    """Manages chat assistant operations and lifecycle."""

    @staticmethod
    def create_chat_assistant(tenant_id: str, assistant_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Create a new chat assistant.

        Args:
            tenant_id: Tenant identifier
            assistant_data: Assistant creation data

        Returns:
            Tuple of (success, result_data or error_info)
        """
        logger.info(f"Creating chat assistant for tenant {tenant_id}")

        try:
            # Validate tenant
            tenant_exists, tenant = TenantService.get_by_id(tenant_id)
            if not tenant_exists:
                return False, {"message": "Tenant not found!"}

            # Process dataset configuration
            dataset_ids = [id for id in assistant_data.get("dataset_ids", []) if id]
            if dataset_ids:
                # Validate dataset access
                is_valid, error_msg = DatasetValidator.validate_dataset_access(dataset_ids, tenant_id)
                if not is_valid:
                    return False, {"message": error_msg}

                # Validate embedding model consistency
                is_valid, error_msg = DatasetValidator.validate_embedding_model_consistency(dataset_ids)
                if not is_valid:
                    return False, {"message": error_msg, "code": settings.RetCode.AUTHENTICATION_ERROR}

            assistant_data["kb_ids"] = dataset_ids

            # Process LLM configuration
            llm_config = assistant_data.get("llm")
            if llm_config:
                is_valid, error_msg, processed_config = LLMConfigurationManager.process_llm_configuration(llm_config, tenant_id)
                if not is_valid:
                    return False, {"message": error_msg}

                assistant_data.update(processed_config)

            # Process prompt configuration
            prompt_config = assistant_data.get("prompt")
            processed_prompt_config = PromptConfigurationManager.process_prompt_configuration(prompt_config)

            # Extract configuration fields
            config_fields = ["similarity_threshold", "vector_similarity_weight", "top_n", "rerank_id", "top_k"]
            for field in config_fields:
                if field in processed_prompt_config:
                    assistant_data[field] = processed_prompt_config.pop(field)

            assistant_data["prompt_config"] = processed_prompt_config

            # Validate prompt parameters
            is_valid, error_msg = PromptConfigurationManager.validate_prompt_parameters(processed_prompt_config)
            if not is_valid:
                return False, {"message": error_msg}

            # Set default values
            assistant_data["id"] = get_uuid()
            assistant_data["description"] = assistant_data.get("description", "A helpful Assistant")
            assistant_data["icon"] = assistant_data.get("avatar", "")
            assistant_data["top_n"] = assistant_data.get("top_n", 6)
            assistant_data["top_k"] = assistant_data.get("top_k", 1024)
            assistant_data["rerank_id"] = assistant_data.get("rerank_id", "")

            # Validate rerank model
            if assistant_data.get("rerank_id"):
                is_valid, error_msg = LLMConfigurationManager.validate_rerank_model(assistant_data["rerank_id"], tenant_id)
                if not is_valid:
                    return False, {"message": error_msg}

            # Set default LLM if not provided
            if not assistant_data.get("llm_id"):
                assistant_data["llm_id"] = tenant.llm_id

            # Validate required fields
            if not assistant_data.get("name"):
                return False, {"message": "`name` is required."}

            # Check for duplicate names
            if DialogService.query(name=assistant_data["name"], tenant_id=tenant_id, status=StatusEnum.VALID.value):
                return False, {"message": "Duplicated chat name in creating chat."}

            # Prevent tenant_id override
            if assistant_data.get("tenant_id"):
                return False, {"message": "`tenant_id` must not be provided."}

            assistant_data["tenant_id"] = tenant_id

            # Save assistant
            if not DialogService.save(**assistant_data):
                return False, {"message": "Failed to create chat assistant!"}

            # Retrieve created assistant
            success, created_assistant = DialogService.get_by_id(assistant_data["id"])
            if not success:
                return False, {"message": "Failed to create chat assistant!"}

            # Format response
            response_data = ChatAssistantManager._format_assistant_response(created_assistant.to_json(), assistant_data["dataset_ids"])

            logger.info(f"Successfully created chat assistant {assistant_data['id']}")
            return True, response_data

        except Exception as error:
            logger.exception(f"Error creating chat assistant: {error}")
            return False, {"message": f"Internal error: {str(error)}"}

    @staticmethod
    def update_chat_assistant(tenant_id: str, chat_id: str, update_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Update an existing chat assistant.

        Args:
            tenant_id: Tenant identifier
            chat_id: Chat assistant identifier
            update_data: Data to update

        Returns:
            Tuple of (success, error_message)
        """
        logger.info(f"Updating chat assistant {chat_id} for tenant {tenant_id}")

        try:
            # Validate ownership
            if not DialogService.query(tenant_id=tenant_id, id=chat_id, status=StatusEnum.VALID.value):
                return False, "You do not own the chat"

            # Process dataset updates
            dataset_ids = update_data.get("dataset_ids")
            if dataset_ids is not None:
                # Validate dataset access
                is_valid, error_msg = DatasetValidator.validate_dataset_access(dataset_ids, tenant_id)
                if not is_valid:
                    return False, error_msg

                # Validate embedding model consistency
                is_valid, error_msg = DatasetValidator.validate_embedding_model_consistency(dataset_ids)
                if not is_valid:
                    return False, error_msg

                update_data["kb_ids"] = dataset_ids

            # Process LLM configuration updates
            llm_config = update_data.get("llm")
            if llm_config:
                is_valid, error_msg, processed_config = LLMConfigurationManager.process_llm_configuration(llm_config, tenant_id)
                if not is_valid:
                    return False, error_msg

                update_data.update(processed_config)

            # Get existing assistant for prompt merging
            success, existing_assistant = DialogService.get_by_id(chat_id)
            if not success:
                return False, "Chat assistant not found"

            existing_data = existing_assistant.to_json()

            # Process prompt configuration updates
            prompt_config = update_data.get("prompt")
            if prompt_config:
                processed_prompt_config = PromptConfigurationManager.process_prompt_configuration(prompt_config)

                # Extract configuration fields
                config_fields = ["similarity_threshold", "vector_similarity_weight", "top_n", "rerank_id", "top_k"]
                for field in config_fields:
                    if field in processed_prompt_config:
                        update_data[field] = processed_prompt_config.pop(field)

                # Merge with existing prompt config
                existing_data["prompt_config"].update(processed_prompt_config)
                update_data["prompt_config"] = existing_data["prompt_config"]

                # Validate prompt parameters
                is_valid, error_msg = PromptConfigurationManager.validate_prompt_parameters(update_data["prompt_config"])
                if not is_valid:
                    return False, error_msg

            # Merge LLM settings
            if "llm_setting" in update_data:
                existing_data["llm_setting"].update(update_data["llm_setting"])
                update_data["llm_setting"] = existing_data["llm_setting"]

            # Validate rerank model
            if update_data.get("rerank_id"):
                is_valid, error_msg = LLMConfigurationManager.validate_rerank_model(update_data["rerank_id"], tenant_id)
                if not is_valid:
                    return False, error_msg

            # Validate name uniqueness
            if "name" in update_data:
                if not update_data.get("name"):
                    return False, "`name` cannot be empty."

                if update_data["name"].lower() != existing_data["name"].lower() and len(DialogService.query(name=update_data["name"], tenant_id=tenant_id, status=StatusEnum.VALID.value)) > 0:
                    return False, "Duplicated chat name in updating chat."

            # Handle avatar field mapping
            if "avatar" in update_data:
                update_data["icon"] = update_data.pop("avatar")

            # Remove dataset_ids from update data (use kb_ids instead)
            update_data.pop("dataset_ids", None)

            # Perform update
            if not DialogService.update_by_id(chat_id, update_data):
                return False, "Chat assistant not found!"

            logger.info(f"Successfully updated chat assistant {chat_id}")
            return True, None

        except Exception as error:
            logger.exception(f"Error updating chat assistant {chat_id}: {error}")
            return False, f"Internal error: {str(error)}"

    @staticmethod
    def _format_assistant_response(assistant_data: Dict[str, Any], dataset_ids: List[str]) -> Dict[str, Any]:
        """
        Format assistant data for API response.

        Args:
            assistant_data: Raw assistant data
            dataset_ids: List of dataset IDs

        Returns:
            Formatted response data
        """
        # Apply key mapping for response
        key_mapping = {
            "parameters": "variables",
            "prologue": "opener",
            "quote": "show_quote",
            "system": "prompt",
            "rerank_id": "rerank_model",
            "vector_similarity_weight": "keywords_similarity_weight",
            "do_refer": "show_quotation",
        }

        # Rename prompt config keys
        renamed_prompt_config = {}
        for key, value in assistant_data["prompt_config"].items():
            new_key = key_mapping.get(key, key)
            renamed_prompt_config[new_key] = value

        assistant_data["prompt"] = renamed_prompt_config
        del assistant_data["prompt_config"]

        # Add additional fields to prompt
        additional_fields = {
            "similarity_threshold": assistant_data["similarity_threshold"],
            "keywords_similarity_weight": 1 - assistant_data["vector_similarity_weight"],
            "top_n": assistant_data["top_n"],
            "rerank_model": assistant_data["rerank_id"],
        }
        assistant_data["prompt"].update(additional_fields)

        # Remove fields that are now in prompt
        config_fields = ["similarity_threshold", "vector_similarity_weight", "top_n", "rerank_id"]
        for field in config_fields:
            assistant_data.pop(field, None)

        # Rename LLM fields
        assistant_data["llm"] = assistant_data.pop("llm_setting")
        assistant_data["llm"]["model_name"] = assistant_data.pop("llm_id")

        # Remove kb_ids and add dataset_ids
        assistant_data.pop("kb_ids", None)
        assistant_data["dataset_ids"] = dataset_ids

        # Rename icon to avatar
        assistant_data["avatar"] = assistant_data.pop("icon")

        return assistant_data


class ChatAssistantListManager:
    """Manages chat assistant listing operations."""

    @staticmethod
    def list_chat_assistants(tenant_id: str, query_params: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        List chat assistants with filtering and pagination.

        Args:
            tenant_id: Tenant identifier
            query_params: Query parameters for filtering and pagination

        Returns:
            Tuple of (success, assistant_list)
        """
        logger.info(f"Listing chat assistants for tenant {tenant_id}")

        try:
            # Extract query parameters
            assistant_id = query_params.get("id")
            assistant_name = query_params.get("name")

            # Validate specific assistant if requested
            if assistant_id or assistant_name:
                assistant = DialogService.query(id=assistant_id, name=assistant_name, status=StatusEnum.VALID.value, tenant_id=tenant_id)
                if not assistant:
                    return False, []

            # Extract pagination parameters
            page_number = int(query_params.get("page", 1))
            items_per_page = int(query_params.get("page_size", 30))
            order_by = query_params.get("orderby", "create_time")
            is_descending = query_params.get("desc", "true").lower() not in ["false", "0"]

            # Get assistants
            assistants = DialogService.get_list(tenant_id, page_number, items_per_page, order_by, is_descending, assistant_id, assistant_name)

            if not assistants:
                return True, []

            # Process assistants for response
            processed_assistants = []
            for assistant in assistants:
                processed_assistant = ChatAssistantListManager._process_assistant_for_listing(assistant)
                processed_assistants.append(processed_assistant)

            logger.info(f"Successfully listed {len(processed_assistants)} chat assistants")
            return True, processed_assistants

        except Exception as error:
            logger.exception(f"Error listing chat assistants: {error}")
            return False, []

    @staticmethod
    def _process_assistant_for_listing(assistant_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process assistant data for listing response.

        Args:
            assistant_data: Raw assistant data

        Returns:
            Processed assistant data
        """
        # Apply key mapping
        key_mapping = {
            "parameters": "variables",
            "prologue": "opener",
            "quote": "show_quote",
            "system": "prompt",
            "rerank_id": "rerank_model",
            "vector_similarity_weight": "keywords_similarity_weight",
            "do_refer": "show_quotation",
        }

        # Rename prompt config keys
        renamed_prompt_config = {}
        for key, value in assistant_data["prompt_config"].items():
            new_key = key_mapping.get(key, key)
            renamed_prompt_config[new_key] = value

        assistant_data["prompt"] = renamed_prompt_config
        del assistant_data["prompt_config"]

        # Add additional fields to prompt
        additional_fields = {
            "similarity_threshold": assistant_data["similarity_threshold"],
            "keywords_similarity_weight": 1 - assistant_data["vector_similarity_weight"],
            "top_n": assistant_data["top_n"],
            "rerank_model": assistant_data["rerank_id"],
        }
        assistant_data["prompt"].update(additional_fields)

        # Remove fields that are now in prompt
        config_fields = ["similarity_threshold", "vector_similarity_weight", "top_n", "rerank_id"]
        for field in config_fields:
            assistant_data.pop(field, None)

        # Rename LLM fields
        assistant_data["llm"] = assistant_data.pop("llm_setting")
        assistant_data["llm"]["model_name"] = assistant_data.pop("llm_id")

        # Process datasets
        dataset_list = []
        for dataset_id in assistant_data["kb_ids"]:
            knowledge_base = KnowledgebaseService.query(id=dataset_id)
            if not knowledge_base:
                logger.warning(f"Dataset {dataset_id} does not exist.")
                continue
            dataset_list.append(knowledge_base[0].to_json())

        assistant_data.pop("kb_ids", None)
        assistant_data["datasets"] = dataset_list

        # Rename icon to avatar
        assistant_data["avatar"] = assistant_data.pop("icon")

        return assistant_data


class ChatAssistantDeletionManager:
    """Manages chat assistant deletion operations."""

    @staticmethod
    def delete_chat_assistants(tenant_id: str, deletion_request: Optional[Dict[str, Any]]) -> Tuple[int, List[str], List[str]]:
        """
        Delete chat assistants.

        Args:
            tenant_id: Tenant identifier
            deletion_request: Deletion request data

        Returns:
            Tuple of (successful_deletions, errors, duplicate_messages)
        """
        logger.info(f"Deleting chat assistants for tenant {tenant_id}")

        try:
            # Determine assistants to delete
            if deletion_request and deletion_request.get("ids"):
                assistant_ids = deletion_request["ids"]
            else:
                # Delete all assistants
                all_assistants = DialogService.query(tenant_id=tenant_id, status=StatusEnum.VALID.value)
                assistant_ids = [assistant.id for assistant in all_assistants]

            # Check for duplicates
            unique_assistant_ids, duplicate_messages = check_duplicate_ids(assistant_ids, "assistant")

            # Perform deletions
            deletion_errors = []
            successful_deletions = 0

            for assistant_id in unique_assistant_ids:
                if not DialogService.query(tenant_id=tenant_id, id=assistant_id, status=StatusEnum.VALID.value):
                    deletion_errors.append(f"Assistant({assistant_id}) not found.")
                    continue

                # Soft delete by setting status to invalid
                update_data = {"status": StatusEnum.INVALID.value}
                DialogService.update_by_id(assistant_id, update_data)
                successful_deletions += 1

            logger.info(f"Successfully deleted {successful_deletions} chat assistants")
            return successful_deletions, deletion_errors, duplicate_messages

        except Exception as error:
            logger.exception(f"Error deleting chat assistants: {error}")
            return 0, [f"Internal error: {str(error)}"], []


# Route handlers using the new managers
@manager.route("/chats", methods=["POST"])  # noqa: F821
@token_required
def create_chat_assistant_endpoint(tenant_id: str) -> Any:
    """Create a new chat assistant."""
    assistant_data = request.json or {}

    success, result = ChatAssistantManager.create_chat_assistant(tenant_id, assistant_data)

    if success:
        return get_result(data=result)
    else:
        error_code = result.get("code", settings.RetCode.DATA_ERROR)
        return get_error_data_result(result["message"], error_code)


@manager.route("/chats/<chat_id>", methods=["PUT"])  # noqa: F821
@token_required
def update_chat_assistant_endpoint(tenant_id: str, chat_id: str) -> Any:
    """Update an existing chat assistant."""
    update_data = request.json or {}

    # Handle show_quotation field mapping
    if "show_quotation" in update_data:
        update_data["do_refer"] = update_data.pop("show_quotation")

    success, error_message = ChatAssistantManager.update_chat_assistant(tenant_id, chat_id, update_data)

    if success:
        return get_result()
    else:
        return get_error_data_result(error_message)


@manager.route("/chats", methods=["DELETE"])  # noqa: F821
@token_required
def delete_chat_assistants_endpoint(tenant_id: str) -> Any:
    """Delete chat assistants."""
    deletion_request = request.json

    successful_deletions, errors, duplicate_messages = ChatAssistantDeletionManager.delete_chat_assistants(tenant_id, deletion_request)

    if errors:
        if successful_deletions > 0:
            return get_result(data={"success_count": successful_deletions, "errors": errors}, message=f"Partially deleted {successful_deletions} chats with {len(errors)} errors")
        else:
            return get_error_data_result("; ".join(errors))

    if duplicate_messages:
        if successful_deletions > 0:
            return get_result(
                message=f"Partially deleted {successful_deletions} chats with {len(duplicate_messages)} errors", data={"success_count": successful_deletions, "errors": duplicate_messages}
            )
        else:
            return get_error_data_result(";".join(duplicate_messages))

    return get_result()


@manager.route("/chats", methods=["GET"])  # noqa: F821
@token_required
def list_chat_assistants_endpoint(tenant_id: str) -> Any:
    """List chat assistants with filtering and pagination."""
    query_params = {
        "id": request.args.get("id"),
        "name": request.args.get("name"),
        "page": request.args.get("page", 1),
        "page_size": request.args.get("page_size", 30),
        "orderby": request.args.get("orderby", "create_time"),
        "desc": request.args.get("desc", "true"),
    }

    success, assistant_list = ChatAssistantListManager.list_chat_assistants(tenant_id, query_params)

    if success:
        return get_result(data=assistant_list)
    else:
        return get_error_data_result("The chat doesn't exist")
