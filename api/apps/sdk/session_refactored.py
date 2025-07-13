"""
Session Management API Module

This module handles session management operations for both chat assistants and agents.
It provides endpoints for creating, updating, listing, and deleting sessions, as well as
handling chat completions with OpenAI-compatible APIs.

Key Features:
- Session lifecycle management (CRUD operations)
- OpenAI-compatible chat completion endpoints
- Agent session management
- Streaming and non-streaming responses
- Authentication and authorization
- Comprehensive error handling and logging

Author: RAGFlow Team
License: Apache License 2.0
"""

import json
import logging
import re
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import tiktoken
from flask import Response, jsonify, request

from agent.canvas import Canvas
from api import settings
from api.db import LLMType, StatusEnum
from api.db.db_models import APIToken
from api.db.services.api_service import API4ConversationService
from api.db.services.canvas_service import UserCanvasService
from api.db.services.conversation_service import ConversationService, iframe_completion
from api.db.services.conversation_service import completion as rag_completion
from api.db.services.canvas_service import completion as agent_completion, completionOpenAI
from api.db.services.dialog_service import DialogService, ask, chat
from api.db.services.file_service import FileService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.llm_service import LLMBundle
from api.utils import get_uuid
from api.utils.api_utils import get_result, token_required, get_data_openai, get_error_data_result, validate_request, check_duplicate_ids

# Configure module-specific logger
logger = logging.getLogger(__name__)


class SessionError(Exception):
    """Custom exception for session-related errors."""

    def __init__(self, message: str, code: int = 500):
        self.message = message
        self.code = code
        super().__init__(self.message)


class ChatSessionManager:
    """Manages chat session operations and lifecycle."""

    @staticmethod
    def create_chat_session(tenant_id: str, chat_id: str, session_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Create a new chat session.

        Args:
            tenant_id: The tenant identifier
            chat_id: The chat/dialog identifier
            session_data: Session creation data

        Returns:
            Tuple of (success, session_dict or error_info)
        """
        logger.info(f"Creating chat session for tenant {tenant_id}, chat {chat_id}")

        try:
            # Validate dialog ownership
            dialog_query_result = DialogService.query(tenant_id=tenant_id, id=chat_id, status=StatusEnum.VALID.value)

            if not dialog_query_result:
                logger.warning(f"Dialog {chat_id} not found or not owned by tenant {tenant_id}")
                return False, {"message": "You do not own the assistant."}

            dialog = dialog_query_result[0]

            # Prepare conversation data
            conversation_data = {
                "id": get_uuid(),
                "dialog_id": chat_id,
                "name": session_data.get("name", "New session"),
                "message": [{"role": "assistant", "content": dialog.prompt_config.get("prologue", "")}],
                "user_id": session_data.get("user_id", ""),
            }

            # Validate session name
            if not conversation_data.get("name"):
                return False, {"message": "`name` cannot be empty."}

            # Save conversation
            if not ConversationService.save(**conversation_data):
                logger.error(f"Failed to save conversation for session {conversation_data['id']}")
                return False, {"message": "Failed to create session!"}

            # Retrieve created conversation
            success, created_conversation = ConversationService.get_by_id(conversation_data["id"])
            if not success:
                logger.error(f"Failed to retrieve created session {conversation_data['id']}")
                return False, {"message": "Failed to create session!"}

            # Format response
            session_dict = created_conversation.to_dict()
            session_dict["messages"] = session_dict.pop("message")
            session_dict["chat_id"] = session_dict.pop("dialog_id")
            session_dict.pop("reference", None)

            logger.info(f"Successfully created chat session {session_dict['id']}")
            return True, session_dict

        except Exception as error:
            logger.exception(f"Error creating chat session: {error}")
            return False, {"message": f"Internal error: {str(error)}"}

    @staticmethod
    def update_chat_session(tenant_id: str, chat_id: str, session_id: str, update_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Update an existing chat session.

        Args:
            tenant_id: The tenant identifier
            chat_id: The chat/dialog identifier
            session_id: The session identifier
            update_data: Data to update

        Returns:
            Tuple of (success, error_message)
        """
        logger.info(f"Updating chat session {session_id} for tenant {tenant_id}")

        try:
            # Validate session existence
            existing_conversation = ConversationService.query(id=session_id, dialog_id=chat_id)
            if not existing_conversation:
                return False, "Session does not exist"

            # Validate dialog ownership
            if not DialogService.query(id=chat_id, tenant_id=tenant_id, status=StatusEnum.VALID.value):
                return False, "You do not own the session"

            # Validate restricted fields
            restricted_fields = ["message", "messages", "reference"]
            for field in restricted_fields:
                if field in update_data:
                    return False, f"`{field}` cannot be changed"

            # Validate name field
            if "name" in update_data and not update_data.get("name"):
                return False, "`name` cannot be empty."

            # Perform update
            if not ConversationService.update_by_id(session_id, update_data):
                return False, "Session update failed"

            logger.info(f"Successfully updated chat session {session_id}")
            return True, None

        except Exception as error:
            logger.exception(f"Error updating chat session {session_id}: {error}")
            return False, f"Internal error: {str(error)}"


class AgentSessionManager:
    """Manages agent session operations and lifecycle."""

    @staticmethod
    def create_agent_session(tenant_id: str, agent_id: str, request_data: Dict[str, Any], files: Dict[str, Any], user_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Create a new agent session.

        Args:
            tenant_id: The tenant identifier
            agent_id: The agent identifier
            request_data: Request data
            files: Uploaded files
            user_id: User identifier

        Returns:
            Tuple of (success, session_dict or error_info)
        """
        logger.info(f"Creating agent session for tenant {tenant_id}, agent {agent_id}")

        try:
            # Validate agent existence and ownership
            agent_exists, canvas_service = UserCanvasService.get_by_id(agent_id)
            if not agent_exists:
                return False, {"message": "Agent not found."}

            if not UserCanvasService.query(user_id=tenant_id, id=agent_id):
                return False, {"message": "You cannot access the agent."}

            # Prepare canvas DSL
            if not isinstance(canvas_service.dsl, str):
                canvas_service.dsl = json.dumps(canvas_service.dsl, ensure_ascii=False)

            # Initialize canvas
            canvas = Canvas(canvas_service.dsl, tenant_id)
            canvas.reset()

            # Process preset parameters
            preset_parameters = canvas.get_preset_param()
            if preset_parameters:
                success, error_message = AgentSessionManager._process_preset_parameters(preset_parameters, request_data, files, user_id)
                if not success:
                    return False, {"message": error_message}

            # Run canvas
            for _ in canvas.run(stream=False):
                pass

            # Create conversation
            canvas_service.dsl = json.loads(str(canvas))
            conversation_data = {
                "id": get_uuid(),
                "dialog_id": canvas_service.id,
                "user_id": user_id,
                "message": [{"role": "assistant", "content": canvas.get_prologue()}],
                "source": "agent",
                "dsl": canvas_service.dsl,
            }

            API4ConversationService.save(**conversation_data)
            conversation_data["agent_id"] = conversation_data.pop("dialog_id")

            logger.info(f"Successfully created agent session {conversation_data['id']}")
            return True, conversation_data

        except Exception as error:
            logger.exception(f"Error creating agent session: {error}")
            return False, {"message": f"Internal error: {str(error)}"}

    @staticmethod
    def _process_preset_parameters(parameters: List[Dict[str, Any]], request_data: Dict[str, Any], files: Dict[str, Any], user_id: str) -> Tuple[bool, Optional[str]]:
        """
        Process preset parameters for agent session creation.

        Args:
            parameters: List of preset parameters
            request_data: Request data
            files: Uploaded files
            user_id: User identifier

        Returns:
            Tuple of (success, error_message)
        """
        for parameter in parameters:
            parameter_key = parameter["key"]
            parameter_type = parameter["type"]
            is_optional = parameter["optional"]

            if not is_optional:
                # Handle required parameters
                if parameter_type == "file":
                    if not files or not files.get(parameter_key):
                        return False, f"`{parameter_key}` with type `{parameter_type}` is required"

                    uploaded_file = files.get(parameter_key)
                    file_content = FileService.parse_docs([uploaded_file], user_id)
                    parameter["value"] = f"{uploaded_file.filename}\n{file_content}"
                else:
                    if not request_data or not request_data.get(parameter_key):
                        return False, f"`{parameter_key}` with type `{parameter_type}` is required"

                    parameter["value"] = request_data[parameter_key]
            else:
                # Handle optional parameters
                if parameter_type == "file":
                    if files and files.get(parameter_key):
                        uploaded_file = files.get(parameter_key)
                        file_content = FileService.parse_docs([uploaded_file], user_id)
                        parameter["value"] = f"{uploaded_file.filename}\n{file_content}"
                    else:
                        parameter.pop("value", None)
                else:
                    if request_data and request_data.get(parameter_key):
                        parameter["value"] = request_data[parameter_key]
                    else:
                        parameter.pop("value", None)

        return True, None


class ChatCompletionHandler:
    """Handles chat completion operations with streaming support."""

    @staticmethod
    def handle_chat_completion(tenant_id: str, chat_id: str, completion_request: Dict[str, Any]) -> Response:
        """
        Handle chat completion request.

        Args:
            tenant_id: The tenant identifier
            chat_id: The chat identifier
            completion_request: Completion request data

        Returns:
            Flask Response object
        """
        logger.info(f"Processing chat completion for tenant {tenant_id}, chat {chat_id}")

        # Validate dialog ownership
        if not DialogService.query(tenant_id=tenant_id, id=chat_id, status=StatusEnum.VALID.value):
            logger.warning(f"Chat {chat_id} not owned by tenant {tenant_id}")
            return get_error_data_result(f"You don't own the chat {chat_id}")

        # Validate session if provided
        session_id = completion_request.get("session_id")
        if session_id:
            if not ConversationService.query(id=session_id, dialog_id=chat_id):
                logger.warning(f"Session {session_id} not owned by chat {chat_id}")
                return get_error_data_result(f"You don't own the session {session_id}")

        # Handle streaming vs non-streaming
        is_streaming = completion_request.get("stream", True)

        if is_streaming:
            return ChatCompletionHandler._create_streaming_response(tenant_id, chat_id, completion_request)
        else:
            return ChatCompletionHandler._create_non_streaming_response(tenant_id, chat_id, completion_request)

    @staticmethod
    def _create_streaming_response(tenant_id: str, chat_id: str, completion_request: Dict[str, Any]) -> Response:
        """Create streaming response for chat completion."""
        response = Response(rag_completion(tenant_id, chat_id, **completion_request), mimetype="text/event-stream")

        # Set streaming headers
        response.headers.add_header("Cache-control", "no-cache")
        response.headers.add_header("Connection", "keep-alive")
        response.headers.add_header("X-Accel-Buffering", "no")
        response.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")

        return response

    @staticmethod
    def _create_non_streaming_response(tenant_id: str, chat_id: str, completion_request: Dict[str, Any]) -> Response:
        """Create non-streaming response for chat completion."""
        completion_result = None
        for result in rag_completion(tenant_id, chat_id, **completion_request):
            completion_result = result
            break

        return get_result(data=completion_result)


class OpenAICompatibilityHandler:
    """Handles OpenAI-compatible API endpoints."""

    @staticmethod
    def handle_openai_chat_completion(tenant_id: str, chat_id: str, openai_request: Dict[str, Any]) -> Response:
        """
        Handle OpenAI-compatible chat completion.

        Args:
            tenant_id: The tenant identifier
            chat_id: The chat identifier
            openai_request: OpenAI-format request data

        Returns:
            Flask Response object
        """
        logger.info(f"Processing OpenAI chat completion for tenant {tenant_id}, chat {chat_id}")

        # Validate messages
        messages = openai_request.get("messages", [])
        if len(messages) < 1:
            return get_error_data_result("You have to provide messages.")

        if messages[-1]["role"] != "user":
            return get_error_data_result("The last message must be from user.")

        # Validate dialog ownership
        dialog_query = DialogService.query(tenant_id=tenant_id, id=chat_id, status=StatusEnum.VALID.value)
        if not dialog_query:
            return get_error_data_result(f"You don't own the chat {chat_id}")

        dialog = dialog_query[0]

        # Process messages
        processed_messages = OpenAICompatibilityHandler._process_openai_messages(messages)

        # Handle streaming vs non-streaming
        is_streaming = openai_request.get("stream", True)

        if is_streaming:
            return OpenAICompatibilityHandler._create_openai_streaming_response(chat_id, dialog, processed_messages, openai_request)
        else:
            return OpenAICompatibilityHandler._create_openai_non_streaming_response(chat_id, dialog, processed_messages, openai_request)

    @staticmethod
    def _process_openai_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process OpenAI messages format."""
        processed_messages = []

        for message in messages:
            if message["role"] == "system":
                continue
            if message["role"] == "assistant" and not processed_messages:
                continue
            processed_messages.append(message)

        return processed_messages

    @staticmethod
    def _create_openai_streaming_response(chat_id: str, dialog: Any, messages: List[Dict[str, Any]], request_data: Dict[str, Any]) -> Response:
        """Create OpenAI-compatible streaming response."""

        def generate_streaming_response() -> Generator[str, None, None]:
            """Generate streaming response chunks."""
            token_count = 0
            answer_buffer = ""
            reasoning_buffer = ""

            base_response = {
                "id": f"chatcmpl-{chat_id}",
                "choices": [{"delta": {"content": "", "role": "assistant", "function_call": None, "tool_calls": None, "reasoning_content": ""}, "finish_reason": None, "index": 0, "logprobs": None}],
                "created": int(time.time()),
                "model": "model",
                "object": "chat.completion.chunk",
                "system_fingerprint": "",
                "usage": None,
            }

            try:
                for response_chunk in chat(dialog, messages, True, toolcall_session=None, tools=None):
                    answer_text = response_chunk["answer"]

                    # Extract reasoning and content
                    reasoning_match = re.search(r"<think>(.*?)</think>", answer_text, flags=re.DOTALL)
                    if reasoning_match:
                        reasoning_part = reasoning_match.group(1)
                        content_part = answer_text[reasoning_match.end() :]
                    else:
                        reasoning_part = ""
                        content_part = answer_text

                    # Calculate incremental content
                    reasoning_incremental = ""
                    if reasoning_part and reasoning_part.startswith(reasoning_buffer):
                        reasoning_incremental = reasoning_part.replace(reasoning_buffer, "", 1)
                        reasoning_buffer = reasoning_part

                    content_incremental = ""
                    if content_part and content_part.startswith(answer_buffer):
                        content_incremental = content_part.replace(answer_buffer, "", 1)
                        answer_buffer = content_part

                    token_count += len(reasoning_incremental) + len(content_incremental)

                    if reasoning_incremental or content_incremental:
                        base_response["choices"][0]["delta"]["reasoning_content"] = reasoning_incremental or None
                        base_response["choices"][0]["delta"]["content"] = content_incremental or None

                        yield f"data:{json.dumps(base_response, ensure_ascii=False)}\n\n"

            except Exception as error:
                logger.exception(f"Error in streaming response: {error}")
                base_response["choices"][0]["delta"]["content"] = f"**ERROR**: {str(error)}"
                yield f"data:{json.dumps(base_response, ensure_ascii=False)}\n\n"

            # Final chunk
            prompt_text = messages[-1]["content"]
            base_response["choices"][0]["delta"]["content"] = None
            base_response["choices"][0]["delta"]["reasoning_content"] = None
            base_response["choices"][0]["finish_reason"] = "stop"
            base_response["usage"] = {"prompt_tokens": len(prompt_text), "completion_tokens": token_count, "total_tokens": len(prompt_text) + token_count}

            yield f"data:{json.dumps(base_response, ensure_ascii=False)}\n\n"
            yield "data:[DONE]\n\n"

        response = Response(generate_streaming_response(), mimetype="text/event-stream")
        response.headers.add_header("Cache-control", "no-cache")
        response.headers.add_header("Connection", "keep-alive")
        response.headers.add_header("X-Accel-Buffering", "no")
        response.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")

        return response

    @staticmethod
    def _create_openai_non_streaming_response(chat_id: str, dialog: Any, messages: List[Dict[str, Any]], request_data: Dict[str, Any]) -> Response:
        """Create OpenAI-compatible non-streaming response."""
        completion_result = None
        for result in chat(dialog, messages, False, toolcall_session=None, tools=None):
            completion_result = result
            break

        content = completion_result["answer"]
        prompt_text = messages[-1]["content"]

        response_data = {
            "id": f"chatcmpl-{chat_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_data.get("model", ""),
            "usage": {
                "prompt_tokens": len(prompt_text),
                "completion_tokens": len(content),
                "total_tokens": len(prompt_text) + len(content),
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "accepted_prediction_tokens": len(content),
                    "rejected_prediction_tokens": 0,
                },
            },
            "choices": [{"message": {"role": "assistant", "content": content}, "logprobs": None, "finish_reason": "stop", "index": 0}],
        }

        return jsonify(response_data)


class SessionListManager:
    """Manages session listing operations."""

    @staticmethod
    def list_chat_sessions(tenant_id: str, chat_id: str, query_params: Dict[str, Any]) -> Response:
        """
        List chat sessions with pagination and filtering.

        Args:
            tenant_id: The tenant identifier
            chat_id: The chat identifier
            query_params: Query parameters for filtering and pagination

        Returns:
            Flask Response object
        """
        logger.info(f"Listing chat sessions for tenant {tenant_id}, chat {chat_id}")

        # Validate dialog ownership
        if not DialogService.query(tenant_id=tenant_id, id=chat_id, status=StatusEnum.VALID.value):
            return get_error_data_result(f"You don't own the assistant {chat_id}.")

        # Extract query parameters
        session_id = query_params.get("id")
        session_name = query_params.get("name")
        page_number = int(query_params.get("page", 1))
        items_per_page = int(query_params.get("page_size", 30))
        order_by = query_params.get("orderby", "create_time")
        user_id = query_params.get("user_id")
        is_descending = query_params.get("desc", "true").lower() not in ["false", "0"]

        # Get conversations
        conversations = ConversationService.get_list(chat_id, page_number, items_per_page, order_by, is_descending, session_id, session_name, user_id)

        if not conversations:
            return get_result(data=[])

        # Process conversations
        processed_conversations = []
        for conversation in conversations:
            processed_conversation = SessionListManager._process_conversation_for_listing(conversation)
            processed_conversations.append(processed_conversation)

        return get_result(data=processed_conversations)

    @staticmethod
    def _process_conversation_for_listing(conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Process conversation data for listing response."""
        # Rename fields
        conversation["messages"] = conversation.pop("message")
        conversation["chat_id"] = conversation.pop("dialog_id")

        # Clean up messages
        for message in conversation["messages"]:
            message.pop("prompt", None)

        # Process references
        if conversation.get("reference"):
            SessionListManager._process_conversation_references(conversation)

        conversation.pop("reference", None)
        return conversation

    @staticmethod
    def _process_conversation_references(conversation: Dict[str, Any]) -> None:
        """Process conversation references for listing."""
        messages = conversation["messages"]
        references = conversation["reference"]

        message_index = 0
        while message_index < len(messages) and message_index < len(references):
            if message_index != 0 and messages[message_index]["role"] != "user":
                if message_index >= len(references):
                    break

                chunk_list = []
                if "chunks" in references[message_index]:
                    chunks = references[message_index]["chunks"]
                    for chunk in chunks:
                        processed_chunk = {
                            "id": chunk.get("chunk_id", chunk.get("id")),
                            "content": chunk.get("content_with_weight", chunk.get("content")),
                            "document_id": chunk.get("doc_id", chunk.get("document_id")),
                            "document_name": chunk.get("docnm_kwd", chunk.get("document_name")),
                            "dataset_id": chunk.get("kb_id", chunk.get("dataset_id")),
                            "image_id": chunk.get("image_id", chunk.get("img_id")),
                            "positions": chunk.get("positions", chunk.get("position_int")),
                        }
                        chunk_list.append(processed_chunk)

                messages[message_index]["reference"] = chunk_list

            message_index += 1


class SessionDeletionManager:
    """Manages session deletion operations."""

    @staticmethod
    def delete_chat_sessions(tenant_id: str, chat_id: str, deletion_request: Optional[Dict[str, Any]]) -> Response:
        """
        Delete chat sessions.

        Args:
            tenant_id: The tenant identifier
            chat_id: The chat identifier
            deletion_request: Deletion request data

        Returns:
            Flask Response object
        """
        logger.info(f"Deleting chat sessions for tenant {tenant_id}, chat {chat_id}")

        # Validate dialog ownership
        if not DialogService.query(id=chat_id, tenant_id=tenant_id, status=StatusEnum.VALID.value):
            return get_error_data_result("You don't own the chat")

        # Determine sessions to delete
        if deletion_request and deletion_request.get("ids"):
            session_ids = deletion_request["ids"]
        else:
            # Delete all sessions
            all_conversations = ConversationService.query(dialog_id=chat_id)
            session_ids = [conv.id for conv in all_conversations]

        # Check for duplicates
        unique_session_ids, duplicate_messages = check_duplicate_ids(session_ids, "session")

        # Perform deletions
        deletion_errors = []
        successful_deletions = 0

        for session_id in unique_session_ids:
            conversation = ConversationService.query(id=session_id, dialog_id=chat_id)
            if not conversation:
                deletion_errors.append(f"The chat doesn't own the session {session_id}")
                continue

            ConversationService.delete_by_id(session_id)
            successful_deletions += 1

        # Build response
        return SessionDeletionManager._build_deletion_response(successful_deletions, deletion_errors, duplicate_messages)

    @staticmethod
    def _build_deletion_response(successful_count: int, errors: List[str], duplicate_messages: List[str]) -> Response:
        """Build response for deletion operation."""
        if errors:
            if successful_count > 0:
                return get_result(data={"success_count": successful_count, "errors": errors}, message=f"Partially deleted {successful_count} sessions with {len(errors)} errors")
            else:
                return get_error_data_result("; ".join(errors))

        if duplicate_messages:
            if successful_count > 0:
                return get_result(
                    message=f"Partially deleted {successful_count} sessions with {len(duplicate_messages)} errors", data={"success_count": successful_count, "errors": duplicate_messages}
                )
            else:
                return get_error_data_result(";".join(duplicate_messages))

        return get_result()


# Route handlers using the new managers
@manager.route("/chats/<chat_id>/sessions", methods=["POST"])  # noqa: F821
@token_required
def create_chat_session_endpoint(tenant_id: str, chat_id: str) -> Response:
    """Create a new chat session."""
    request_data = request.json or {}
    request_data["dialog_id"] = chat_id

    success, result = ChatSessionManager.create_chat_session(tenant_id, chat_id, request_data)

    if success:
        return get_result(data=result)
    else:
        return get_error_data_result(result["message"])


@manager.route("/agents/<agent_id>/sessions", methods=["POST"])  # noqa: F821
@token_required
def create_agent_session_endpoint(tenant_id: str, agent_id: str) -> Response:
    """Create a new agent session."""
    request_data = request.json if request.is_json else request.form.to_dict()
    files = request.files
    user_id = request.args.get("user_id", "")

    success, result = AgentSessionManager.create_agent_session(tenant_id, agent_id, request_data, files, user_id)

    if success:
        return get_result(data=result)
    else:
        return get_error_data_result(result["message"])


@manager.route("/chats/<chat_id>/sessions/<session_id>", methods=["PUT"])  # noqa: F821
@token_required
def update_chat_session_endpoint(tenant_id: str, chat_id: str, session_id: str) -> Response:
    """Update an existing chat session."""
    request_data = request.json or {}
    request_data["dialog_id"] = chat_id

    success, error_message = ChatSessionManager.update_chat_session(tenant_id, chat_id, session_id, request_data)

    if success:
        return get_result()
    else:
        return get_error_data_result(error_message)


@manager.route("/chats/<chat_id>/completions", methods=["POST"])  # noqa: F821
@token_required
def chat_completion_endpoint(tenant_id: str, chat_id: str) -> Response:
    """Handle chat completion requests."""
    completion_request = request.json or {"question": ""}

    if not completion_request.get("session_id"):
        completion_request["question"] = ""

    return ChatCompletionHandler.handle_chat_completion(tenant_id, chat_id, completion_request)


@manager.route("/chats_openai/<chat_id>/chat/completions", methods=["POST"])  # noqa: F821
@validate_request("model", "messages")  # noqa: F821
@token_required
def openai_chat_completion_endpoint(tenant_id: str, chat_id: str) -> Response:
    """
    OpenAI-compatible chat completion endpoint.

    This endpoint provides OpenAI-compatible chat completion functionality,
    supporting both streaming and non-streaming responses.
    """
    openai_request = request.json
    return OpenAICompatibilityHandler.handle_openai_chat_completion(tenant_id, chat_id, openai_request)


@manager.route("/agents_openai/<agent_id>/chat/completions", methods=["POST"])  # noqa: F821
@validate_request("model", "messages")  # noqa: F821
@token_required
def agents_openai_completion_endpoint(tenant_id: str, agent_id: str) -> Response:
    """Handle OpenAI-compatible agent completions."""
    request_data = request.json
    tiktoken_encoder = tiktoken.get_encoding("cl100k_base")

    messages = request_data.get("messages", [])
    if not messages:
        return get_error_data_result("You must provide at least one message.")

    if not UserCanvasService.query(user_id=tenant_id, id=agent_id):
        return get_error_data_result(f"You don't own the agent {agent_id}")

    # Filter and process messages
    filtered_messages = [msg for msg in messages if msg["role"] in ["user", "assistant"]]
    prompt_tokens = sum(len(tiktoken_encoder.encode(msg["content"])) for msg in filtered_messages)

    if not filtered_messages:
        return jsonify(
            get_data_openai(
                id=agent_id,
                content="No valid messages found (user or assistant).",
                finish_reason="stop",
                model=request_data.get("model", ""),
                completion_tokens=len(tiktoken_encoder.encode("No valid messages found (user or assistant).")),
                prompt_tokens=prompt_tokens,
            )
        )

    # Get the last user message
    question = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), "")

    session_id = request_data.get("id", request_data.get("metadata", {}).get("id", ""))

    if request_data.get("stream", True):
        return Response(completionOpenAI(tenant_id, agent_id, question, session_id=session_id, stream=True), mimetype="text/event-stream")
    else:
        response = next(completionOpenAI(tenant_id, agent_id, question, session_id=session_id, stream=False))
        return jsonify(response)


@manager.route("/agents/<agent_id>/completions", methods=["POST"])  # noqa: F821
@token_required
def agent_completions_endpoint(tenant_id: str, agent_id: str) -> Response:
    """Handle agent completion requests."""
    request_data = request.json

    canvas_services = UserCanvasService.query(user_id=tenant_id, id=agent_id)
    if not canvas_services:
        return get_error_data_result(f"You don't own the agent {agent_id}")

    # Handle session synchronization
    if request_data.get("session_id"):
        canvas_dsl = canvas_services[0].dsl
        if not isinstance(canvas_dsl, str):
            canvas_dsl = json.dumps(canvas_dsl)

        conversation = API4ConversationService.query(id=request_data["session_id"], dialog_id=agent_id)
        if not conversation:
            return get_error_data_result(f"You don't own the session {request_data['session_id']}")

        # Synchronize DSL if requested
        sync_dsl = request_data.get("sync_dsl", False)
        if sync_dsl and canvas_services[0].update_time > conversation[0].update_time:
            current_dsl = conversation[0].dsl
            new_dsl = json.loads(canvas_dsl)
            state_fields = ["history", "messages", "path", "reference"]
            states = {field: current_dsl.get(field, []) for field in state_fields}
            current_dsl.update(new_dsl)
            current_dsl.update(states)
            API4ConversationService.update_by_id(request_data["session_id"], {"dsl": current_dsl})
    else:
        request_data["question"] = ""

    # Handle streaming vs non-streaming
    if request_data.get("stream", True):
        response = Response(agent_completion(tenant_id, agent_id, **request_data), mimetype="text/event-stream")
        response.headers.add_header("Cache-control", "no-cache")
        response.headers.add_header("Connection", "keep-alive")
        response.headers.add_header("X-Accel-Buffering", "no")
        response.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
        return response

    try:
        for answer in agent_completion(tenant_id, agent_id, **request_data):
            return get_result(data=answer)
    except Exception as error:
        return get_error_data_result(str(error))


@manager.route("/chats/<chat_id>/sessions", methods=["GET"])  # noqa: F821
@token_required
def list_chat_sessions_endpoint(tenant_id: str, chat_id: str) -> Response:
    """List chat sessions with pagination and filtering."""
    query_params = {
        "id": request.args.get("id"),
        "name": request.args.get("name"),
        "page": request.args.get("page", 1),
        "page_size": request.args.get("page_size", 30),
        "orderby": request.args.get("orderby", "create_time"),
        "user_id": request.args.get("user_id"),
        "desc": request.args.get("desc", "true"),
    }

    return SessionListManager.list_chat_sessions(tenant_id, chat_id, query_params)


@manager.route("/agents/<agent_id>/sessions", methods=["GET"])  # noqa: F821
@token_required
def list_agent_sessions_endpoint(tenant_id: str, agent_id: str) -> Response:
    """List agent sessions with pagination and filtering."""
    if not UserCanvasService.query(user_id=tenant_id, id=agent_id):
        return get_error_data_result(f"You don't own the agent {agent_id}.")

    # Extract query parameters
    session_id = request.args.get("id")
    user_id = request.args.get("user_id")
    page_number = int(request.args.get("page", 1))
    items_per_page = int(request.args.get("page_size", 30))
    order_by = request.args.get("orderby", "update_time")
    is_descending = request.args.get("desc", "true").lower() not in ["false", "0"]
    include_dsl = request.args.get("dsl", "true").lower() not in ["false", "0"]

    # Get conversations
    conversations = API4ConversationService.get_list(agent_id, tenant_id, page_number, items_per_page, order_by, is_descending, session_id, user_id, include_dsl)

    if not conversations:
        return get_result(data=[])

    # Process conversations
    processed_conversations = []
    for conversation in conversations:
        processed_conversation = SessionListManager._process_agent_conversation_for_listing(conversation)
        processed_conversations.append(processed_conversation)

    return get_result(data=processed_conversations)


@manager.route("/chats/<chat_id>/sessions", methods=["DELETE"])  # noqa: F821
@token_required
def delete_chat_sessions_endpoint(tenant_id: str, chat_id: str) -> Response:
    """Delete chat sessions."""
    deletion_request = request.json
    return SessionDeletionManager.delete_chat_sessions(tenant_id, chat_id, deletion_request)


@manager.route("/agents/<agent_id>/sessions", methods=["DELETE"])  # noqa: F821
@token_required
def delete_agent_sessions_endpoint(tenant_id: str, agent_id: str) -> Response:
    """Delete agent sessions."""
    deletion_request = request.json

    canvas_services = UserCanvasService.query(user_id=tenant_id, id=agent_id)
    if not canvas_services:
        return get_error_data_result(f"You don't own the agent {agent_id}")

    conversations = API4ConversationService.query(dialog_id=agent_id)
    if not conversations:
        return get_error_data_result(f"Agent {agent_id} has no sessions")

    # Determine sessions to delete
    if deletion_request and deletion_request.get("ids"):
        session_ids = deletion_request["ids"]
    else:
        session_ids = [conv.id for conv in conversations]

    # Check for duplicates
    unique_session_ids, duplicate_messages = check_duplicate_ids(session_ids, "session")

    # Perform deletions
    deletion_errors = []
    successful_deletions = 0

    for session_id in unique_session_ids:
        conversation = API4ConversationService.query(id=session_id, dialog_id=agent_id)
        if not conversation:
            deletion_errors.append(f"The agent doesn't own the session {session_id}")
            continue

        API4ConversationService.delete_by_id(session_id)
        successful_deletions += 1

    # Build response
    return SessionDeletionManager._build_deletion_response(successful_deletions, deletion_errors, duplicate_messages)


@manager.route("/sessions/ask", methods=["POST"])  # noqa: F821
@token_required
def ask_about_datasets_endpoint(tenant_id: str) -> Response:
    """Ask questions about specific datasets."""
    request_data = request.json

    # Validate required fields
    if not request_data.get("question"):
        return get_error_data_result("`question` is required.")

    if not request_data.get("dataset_ids"):
        return get_error_data_result("`dataset_ids` is required.")

    if not isinstance(request_data.get("dataset_ids"), list):
        return get_error_data_result("`dataset_ids` should be a list.")

    # Validate dataset access
    dataset_ids = request_data.pop("dataset_ids")
    request_data["kb_ids"] = dataset_ids

    for dataset_id in dataset_ids:
        if not KnowledgebaseService.accessible(dataset_id, tenant_id):
            return get_error_data_result(f"You don't own the dataset {dataset_id}.")

        knowledge_bases = KnowledgebaseService.query(id=dataset_id)
        if knowledge_bases and knowledge_bases[0].chunk_num == 0:
            return get_error_data_result(f"The dataset {dataset_id} doesn't have parsed files")

    def generate_streaming_response():
        """Generate streaming response for dataset questions."""
        try:
            for answer in ask(request_data["question"], request_data["kb_ids"], tenant_id):
                yield "data:" + json.dumps({"code": 0, "message": "", "data": answer}, ensure_ascii=False) + "\n\n"
        except Exception as error:
            logger.exception(f"Error in ask endpoint: {error}")
            yield "data:" + json.dumps({"code": 500, "message": str(error), "data": {"answer": f"**ERROR**: {str(error)}", "reference": []}}, ensure_ascii=False) + "\n\n"

        yield "data:" + json.dumps({"code": 0, "message": "", "data": True}, ensure_ascii=False) + "\n\n"

    response = Response(generate_streaming_response(), mimetype="text/event-stream")
    response.headers.add_header("Cache-control", "no-cache")
    response.headers.add_header("Connection", "keep-alive")
    response.headers.add_header("X-Accel-Buffering", "no")
    response.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")

    return response


@manager.route("/sessions/related_questions", methods=["POST"])  # noqa: F821
@token_required
def generate_related_questions_endpoint(tenant_id: str) -> Response:
    """Generate related questions based on user input."""
    request_data = request.json

    if not request_data.get("question"):
        return get_error_data_result("`question` is required.")

    question = request_data["question"]
    chat_model = LLMBundle(tenant_id, LLMType.CHAT)

    prompt_template = """
Objective: To generate search terms related to the user's search keywords, helping users find more valuable information.
Instructions:
 - Based on the keywords provided by the user, generate 5-10 related search terms.
 - Each search term should be directly or indirectly related to the keyword, guiding the user to find more valuable information.
 - Use common, general terms as much as possible, avoiding obscure words or technical jargon.
 - Keep the term length between 2-4 words, concise and clear.
 - DO NOT translate, use the language of the original keywords.

### Example:
Keywords: Chinese football
Related search terms:
1. Current status of Chinese football
2. Reform of Chinese football
3. Youth training of Chinese football
4. Chinese football in the Asian Cup
5. Chinese football in the World Cup

Reason:
 - When searching, users often only use one or two keywords, making it difficult to fully express their information needs.
 - Generating related search terms can help users dig deeper into relevant information and improve search efficiency.
 - At the same time, related terms can also help search engines better understand user needs and return more accurate search results.

"""

    try:
        response = chat_model.chat(
            prompt_template,
            [
                {
                    "role": "user",
                    "content": f"""
Keywords: {question}
Related search terms:
    """,
                }
            ],
            {"temperature": 0.9},
        )

        # Extract numbered items from response
        related_questions = [re.sub(r"^[0-9]\. ", "", line) for line in response.split("\n") if re.match(r"^[0-9]\. ", line)]

        return get_result(data=related_questions)

    except Exception as error:
        logger.exception(f"Error generating related questions: {error}")
        return get_error_data_result(f"Failed to generate related questions: {str(error)}")


@manager.route("/chatbots/<dialog_id>/completions", methods=["POST"])  # noqa: F821
def chatbot_completions_endpoint(dialog_id: str) -> Response:
    """Handle chatbot completions with API key authentication."""
    request_data = request.json

    # Validate API key
    authorization_header = request.headers.get("Authorization", "")
    auth_parts = authorization_header.split()

    if len(auth_parts) != 2:
        return get_error_data_result("Authorization is not valid!")

    token = auth_parts[1]
    api_tokens = APIToken.query(beta=token)

    if not api_tokens:
        return get_error_data_result("Authentication error: API key is invalid!")

    # Set default quote parameter
    if "quote" not in request_data:
        request_data["quote"] = False

    # Handle streaming vs non-streaming
    if request_data.get("stream", True):
        response = Response(iframe_completion(dialog_id, **request_data), mimetype="text/event-stream")
        response.headers.add_header("Cache-control", "no-cache")
        response.headers.add_header("Connection", "keep-alive")
        response.headers.add_header("X-Accel-Buffering", "no")
        response.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
        return response

    for answer in iframe_completion(dialog_id, **request_data):
        return get_result(data=answer)


@manager.route("/agentbots/<agent_id>/completions", methods=["POST"])  # noqa: F821
def agent_bot_completions_endpoint(agent_id: str) -> Response:
    """Handle agent bot completions with API key authentication."""
    request_data = request.json

    # Validate API key
    authorization_header = request.headers.get("Authorization", "")
    auth_parts = authorization_header.split()

    if len(auth_parts) != 2:
        return get_error_data_result("Authorization is not valid!")

    token = auth_parts[1]
    api_tokens = APIToken.query(beta=token)

    if not api_tokens:
        return get_error_data_result("Authentication error: API key is invalid!")

    tenant_id = api_tokens[0].tenant_id

    # Set default quote parameter
    if "quote" not in request_data:
        request_data["quote"] = False

    # Handle streaming vs non-streaming
    if request_data.get("stream", True):
        response = Response(agent_completion(tenant_id, agent_id, **request_data), mimetype="text/event-stream")
        response.headers.add_header("Cache-control", "no-cache")
        response.headers.add_header("Connection", "keep-alive")
        response.headers.add_header("X-Accel-Buffering", "no")
        response.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
        return response

    for answer in agent_completion(tenant_id, agent_id, **request_data):
        return get_result(data=answer)


# Add missing method to SessionListManager
def _process_agent_conversation_for_listing(conversation: Dict[str, Any]) -> Dict[str, Any]:
    """Process agent conversation data for listing response."""
    # Rename fields
    conversation["messages"] = conversation.pop("message")
    conversation["agent_id"] = conversation.pop("dialog_id")

    # Clean up messages
    for message in conversation["messages"]:
        message.pop("prompt", None)

    # Process references for agent conversations
    if conversation.get("reference"):
        messages = conversation["messages"]
        references = conversation["reference"]

        message_index = 0
        chunk_index = 0

        while message_index < len(messages):
            if message_index != 0 and messages[message_index]["role"] != "user":
                chunk_list = []
                if chunk_index < len(references) and "chunks" in references[chunk_index]:
                    chunks = references[chunk_index]["chunks"]
                    for chunk in chunks:
                        processed_chunk = {
                            "id": chunk.get("chunk_id", chunk.get("id")),
                            "content": chunk.get("content_with_weight", chunk.get("content")),
                            "document_id": chunk.get("doc_id", chunk.get("document_id")),
                            "document_name": chunk.get("docnm_kwd", chunk.get("document_name")),
                            "dataset_id": chunk.get("kb_id", chunk.get("dataset_id")),
                            "image_id": chunk.get("image_id", chunk.get("img_id")),
                            "positions": chunk.get("positions", chunk.get("position_int")),
                        }
                        chunk_list.append(processed_chunk)

                chunk_index += 1
                messages[message_index]["reference"] = chunk_list

            message_index += 1

    conversation.pop("reference", None)
    return conversation


# Add the missing method to SessionListManager class
SessionListManager._process_agent_conversation_for_listing = staticmethod(_process_agent_conversation_for_listing)
