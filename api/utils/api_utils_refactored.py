"""
API Utilities Module

This module provides core utilities for API request handling, authentication,
validation, and response formatting. It includes decorators for token validation,
request validation, error handling utilities, and helper functions for data processing.

Author: RAGFlow Team
License: Apache License 2.0
"""

import functools
import json
import logging
import random
import time
from base64 import b64encode
from copy import deepcopy
from functools import wraps
from hmac import HMAC
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote, urlencode
from uuid import uuid1

import requests
from flask import (
    Response,
    jsonify,
    make_response,
    send_file,
)
from flask import (
    request as flask_request,
)
from itsdangerous import URLSafeTimedSerializer
from peewee import OperationalError
from werkzeug.http import HTTP_STATUS_CODES

from api import settings
from api.constants import REQUEST_MAX_WAIT_SEC, REQUEST_WAIT_SEC
from api.db.db_models import APIToken
from api.db.services.llm_service import LLMService, TenantLLMService
from api.utils import CustomJSONEncoder, get_uuid, json_dumps

# Configure logging for this module
logger = logging.getLogger(__name__)

# Configure requests to use custom JSON encoder
requests.models.complexjson.dumps = functools.partial(json.dumps, cls=CustomJSONEncoder)


class APIError(Exception):
    """Custom exception for API-related errors."""

    def __init__(self, message: str, code: int = settings.RetCode.EXCEPTION_ERROR):
        self.message = message
        self.code = code
        super().__init__(self.message)


class AuthenticationManager:
    """Handles authentication-related operations."""

    @staticmethod
    def validate_api_token(authorization_header: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate API token from authorization header.

        Args:
            authorization_header: The Authorization header value

        Returns:
            Tuple of (is_valid, tenant_id, error_message)
        """
        if not authorization_header:
            return False, None, "`Authorization` header is required"

        auth_parts = authorization_header.split()
        if len(auth_parts) < 2:
            return False, None, "Invalid authorization format. Expected: 'Bearer <token>'"

        token = auth_parts[1]
        api_tokens = APIToken.query(token=token)

        if not api_tokens:
            return False, None, "Invalid API token"

        return True, api_tokens[0].tenant_id, None

    @staticmethod
    def generate_confirmation_token(tenant_id: str) -> str:
        """
        Generate a confirmation token for the given tenant.

        Args:
            tenant_id: The tenant identifier

        Returns:
            Generated confirmation token
        """
        serializer = URLSafeTimedSerializer(tenant_id)
        return "ragflow-" + serializer.dumps(get_uuid(), salt=tenant_id)[2:34]


class RequestValidator:
    """Handles request validation operations."""

    @staticmethod
    def validate_required_fields(request_data: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate that all required fields are present in request data.

        Args:
            request_data: The request data dictionary
            required_fields: List of required field names

        Returns:
            Tuple of (is_valid, missing_fields)
        """
        missing_fields = []
        for field in required_fields:
            if field not in request_data or request_data[field] is None:
                missing_fields.append(field)

        return len(missing_fields) == 0, missing_fields

    @staticmethod
    def validate_field_values(request_data: Dict[str, Any], field_constraints: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate field values against constraints.

        Args:
            request_data: The request data dictionary
            field_constraints: Dictionary mapping field names to allowed values

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        for field_name, allowed_values in field_constraints.items():
            field_value = request_data.get(field_name)
            if field_value is None:
                continue

            if isinstance(allowed_values, (tuple, list)):
                if field_value not in allowed_values:
                    errors.append(f"{field_name} must be one of: {allowed_values}")
            elif field_value != allowed_values:
                errors.append(f"{field_name} must be: {allowed_values}")

        return len(errors) == 0, errors


class ResponseBuilder:
    """Handles API response building and formatting."""

    @staticmethod
    def build_success_response(data: Any = None, message: str = "success") -> Response:
        """
        Build a successful API response.

        Args:
            data: Response data
            message: Success message

        Returns:
            Flask Response object
        """
        response_data = {"code": settings.RetCode.SUCCESS}

        if data is not None:
            response_data["data"] = data
        if message != "success":
            response_data["message"] = message

        return jsonify(response_data)

    @staticmethod
    def build_error_response(message: str, code: int = settings.RetCode.DATA_ERROR) -> Response:
        """
        Build an error API response.

        Args:
            message: Error message
            code: Error code

        Returns:
            Flask Response object
        """
        logger.error(f"API Error - Code: {code}, Message: {message}")
        return jsonify({"code": code, "message": message})

    @staticmethod
    def build_openai_compatible_response(
        request_id: str, content: str, model: str, prompt_tokens: int = 0, completion_tokens: int = 0, finish_reason: str = "stop", object_type: str = "chat.completion"
    ) -> Dict[str, Any]:
        """
        Build OpenAI-compatible response format.

        Args:
            request_id: Unique request identifier
            content: Response content
            model: Model name used
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            finish_reason: Reason for completion
            object_type: Type of response object

        Returns:
            OpenAI-compatible response dictionary
        """
        total_tokens = prompt_tokens + completion_tokens

        return {
            "id": f"{request_id}",
            "object": object_type,
            "created": int(time.time()),
            "model": model,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "completion_tokens_details": {"reasoning_tokens": 0, "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0},
            },
            "choices": [{"message": {"role": "assistant", "content": content}, "logprobs": None, "finish_reason": finish_reason, "index": 0}],
        }


class HTTPClientManager:
    """Manages HTTP client operations with authentication."""

    @staticmethod
    def make_authenticated_request(**kwargs) -> requests.Response:
        """
        Make an authenticated HTTP request.

        Args:
            **kwargs: Request parameters

        Returns:
            Response object
        """
        session = requests.Session()
        stream = kwargs.pop("stream", session.stream)
        timeout = kwargs.pop("timeout", None)

        # Normalize headers
        kwargs["headers"] = {key.replace("_", "-").upper(): value for key, value in kwargs.get("headers", {}).items()}

        prepared_request = requests.Request(**kwargs).prepare()

        # Add authentication if configured
        if settings.CLIENT_AUTHENTICATION and settings.HTTP_APP_KEY and settings.SECRET_KEY:
            HTTPClientManager._add_authentication_headers(prepared_request, kwargs)

        return session.send(prepared_request, stream=stream, timeout=timeout)

    @staticmethod
    def _add_authentication_headers(prepared_request: requests.PreparedRequest, original_kwargs: Dict[str, Any]) -> None:
        """Add authentication headers to the prepared request."""
        timestamp = str(round(time.time() * 1000))
        nonce = str(uuid1())

        signature_components = [
            timestamp.encode("ascii"),
            nonce.encode("ascii"),
            settings.HTTP_APP_KEY.encode("ascii"),
            prepared_request.path_url.encode("ascii"),
            prepared_request.body if original_kwargs.get("json") else b"",
        ]

        # Add form data if present
        if original_kwargs.get("data") and isinstance(original_kwargs["data"], dict):
            form_data = urlencode(sorted(original_kwargs["data"].items()), quote_via=quote, safe="-._~").encode("ascii")
            signature_components.append(form_data)
        else:
            signature_components.append(b"")

        signature = b64encode(
            HMAC(
                settings.SECRET_KEY.encode("ascii"),
                b"\n".join(signature_components),
                "sha1",
            ).digest()
        ).decode("ascii")

        prepared_request.headers.update(
            {
                "TIMESTAMP": timestamp,
                "NONCE": nonce,
                "APP-KEY": settings.HTTP_APP_KEY,
                "SIGNATURE": signature,
            }
        )


class RetryManager:
    """Handles retry logic with exponential backoff."""

    @staticmethod
    def calculate_backoff_interval(retry_count: int, use_full_jitter: bool = False) -> float:
        """
        Calculate exponential backoff wait time.

        Args:
            retry_count: Number of retries attempted
            use_full_jitter: Whether to use full jitter

        Returns:
            Wait time in seconds
        """
        base_wait_time = min(REQUEST_MAX_WAIT_SEC, REQUEST_WAIT_SEC * (2**retry_count))

        if use_full_jitter:
            base_wait_time = random.randrange(int(base_wait_time) + 1)

        return max(0, base_wait_time)


class DataProcessor:
    """Handles data processing and transformation operations."""

    @staticmethod
    def check_duplicate_identifiers(identifiers: List[str], identifier_type: str = "item") -> Tuple[List[str], List[str]]:
        """
        Check for duplicate identifiers and return unique ones with error messages.

        Args:
            identifiers: List of identifiers to check
            identifier_type: Type of identifier for error messages

        Returns:
            Tuple of (unique_identifiers, error_messages)
        """
        identifier_counts = {}
        duplicate_errors = []

        # Count occurrences
        for identifier in identifiers:
            identifier_counts[identifier] = identifier_counts.get(identifier, 0) + 1

        # Find duplicates
        for identifier, count in identifier_counts.items():
            if count > 1:
                duplicate_errors.append(f"Duplicate {identifier_type} ID: {identifier}")

        return list(set(identifiers)), duplicate_errors

    @staticmethod
    def deep_merge_dictionaries(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two dictionaries with priority to override values.

        Args:
            base_dict: Base dictionary with default values
            override_dict: Dictionary with overriding values

        Returns:
            Merged dictionary
        """
        merged_result = deepcopy(base_dict)
        merge_stack = [(merged_result, override_dict)]

        while merge_stack:
            current_base, current_override = merge_stack.pop()

            for key, value in current_override.items():
                if key in current_base and isinstance(value, dict) and isinstance(current_base[key], dict):
                    merge_stack.append((current_base[key], value))
                else:
                    current_base[key] = value

        return merged_result

    @staticmethod
    def remap_dictionary_keys(source_data: Dict[str, Any], key_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Transform dictionary keys using a mapping schema.

        Args:
            source_data: Original dictionary to process
            key_mapping: Custom key transformation rules

        Returns:
            Dictionary with transformed keys
        """
        default_mapping = {
            "chunk_num": "chunk_count",
            "doc_num": "document_count",
            "parser_id": "chunk_method",
            "embd_id": "embedding_model",
        }

        mapping = key_mapping or default_mapping
        transformed_data = {}

        for original_key, value in source_data.items():
            new_key = mapping.get(original_key, original_key)
            transformed_data[new_key] = value

        return transformed_data


class EmbeddingModelValidator:
    """Validates embedding model availability and authorization."""

    @staticmethod
    def verify_model_availability(embedding_model_id: str, tenant_id: str) -> Tuple[bool, Optional[Response]]:
        """
        Verify availability of an embedding model for a specific tenant.

        Args:
            embedding_model_id: Unique identifier for the embedding model
            tenant_id: Tenant identifier for access control

        Returns:
            Tuple of (is_available, error_response)
        """
        try:
            model_name, model_factory = TenantLLMService.split_model_name_and_factory(embedding_model_id)

            # Check if model exists in LLM service
            model_exists_in_service = bool(LLMService.query(llm_name=model_name, fid=model_factory, model_type="embedding"))

            # Check tenant-specific models
            tenant_models = TenantLLMService.get_my_llms(tenant_id=tenant_id)
            is_tenant_authorized = any(model["llm_name"] == model_name and model["llm_factory"] == model_factory and model["model_type"] == "embedding" for model in tenant_models)

            # Check built-in models
            is_builtin_model = embedding_model_id in settings.BUILTIN_EMBEDDING_MODELS

            # Validate model existence
            if not (is_builtin_model or is_tenant_authorized or model_exists_in_service):
                error_response = ResponseBuilder.build_error_response(f"Unsupported embedding model: {embedding_model_id}", settings.RetCode.ARGUMENT_ERROR)
                return False, error_response

            # Validate authorization
            if not (is_builtin_model or is_tenant_authorized):
                error_response = ResponseBuilder.build_error_response(f"Unauthorized access to embedding model: {embedding_model_id}", settings.RetCode.ARGUMENT_ERROR)
                return False, error_response

        except OperationalError as database_error:
            logger.exception(f"Database error during model validation: {database_error}")
            error_response = ResponseBuilder.build_error_response("Database operation failed during model validation")
            return False, error_response

        return True, None


class FileManager:
    """Handles file operations for API responses."""

    @staticmethod
    def send_data_as_file(data: Union[str, bytes, Dict[str, Any]], filename: str) -> Response:
        """
        Send data as a downloadable file.

        Args:
            data: Data to send (string, bytes, or JSON-serializable object)
            filename: Name of the file

        Returns:
            Flask Response for file download
        """
        if not isinstance(data, (str, bytes)):
            data = json_dumps(data)

        if isinstance(data, str):
            data = data.encode("utf-8")

        file_buffer = BytesIO()
        file_buffer.write(data)
        file_buffer.seek(0)

        return send_file(file_buffer, as_attachment=True, attachment_filename=filename)


class ConfigurationManager:
    """Manages parser and configuration settings."""

    @staticmethod
    def get_parser_configuration(chunk_method: str, custom_config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Get parser configuration based on chunk method.

        Args:
            chunk_method: The chunking method to use
            custom_config: Custom configuration to override defaults

        Returns:
            Parser configuration dictionary
        """
        if custom_config:
            return custom_config

        if not chunk_method:
            chunk_method = "naive"

        default_configurations = {
            "naive": {"chunk_token_num": 512, "delimiter": r"\n", "html4excel": False, "layout_recognize": "DeepDOC", "raptor": {"use_raptor": False}},
            "qa": {"raptor": {"use_raptor": False}},
            "tag": None,
            "resume": None,
            "manual": {"raptor": {"use_raptor": False}},
            "table": None,
            "paper": {"raptor": {"use_raptor": False}},
            "book": {"raptor": {"use_raptor": False}},
            "laws": {"raptor": {"use_raptor": False}},
            "presentation": {"raptor": {"use_raptor": False}},
            "one": None,
            "knowledge_graph": {"chunk_token_num": 8192, "delimiter": r"\n", "entity_types": ["organization", "person", "location", "event", "time"]},
            "email": None,
            "picture": None,
        }

        return default_configurations.get(chunk_method)


# Decorator Functions
def token_required(function):
    """
    Decorator to require valid API token for endpoint access.

    Args:
        function: The function to decorate

    Returns:
        Decorated function with token validation
    """

    @wraps(function)
    def decorated_function(*args, **kwargs):
        authorization_header = flask_request.headers.get("Authorization")

        is_valid, tenant_id, error_message = AuthenticationManager.validate_api_token(authorization_header)

        if not is_valid:
            return ResponseBuilder.build_error_response(error_message, settings.RetCode.AUTHENTICATION_ERROR)

        kwargs["tenant_id"] = tenant_id
        return function(*args, **kwargs)

    return decorated_function


def validate_request(*required_args, **field_constraints):
    """
    Decorator to validate request parameters.

    Args:
        *required_args: Required field names
        **field_constraints: Field value constraints

    Returns:
        Decorated function with request validation
    """

    def decorator(function):
        @wraps(function)
        def decorated_function(*args, **kwargs):
            request_data = flask_request.json or flask_request.form.to_dict()

            # Validate required fields
            is_valid, missing_fields = RequestValidator.validate_required_fields(request_data, list(required_args))

            if not is_valid:
                error_message = f"Required fields missing: {', '.join(missing_fields)}"
                return ResponseBuilder.build_error_response(error_message, settings.RetCode.ARGUMENT_ERROR)

            # Validate field constraints
            is_valid, constraint_errors = RequestValidator.validate_field_values(request_data, field_constraints)

            if not is_valid:
                error_message = "; ".join(constraint_errors)
                return ResponseBuilder.build_error_response(error_message, settings.RetCode.ARGUMENT_ERROR)

            return function(*args, **kwargs)

        return decorated_function

    return decorator


def not_allowed_parameters(*forbidden_params):
    """
    Decorator to prevent certain parameters in requests.

    Args:
        *forbidden_params: Parameter names that are not allowed

    Returns:
        Decorated function with parameter restriction
    """

    def decorator(function):
        @wraps(function)
        def decorated_function(*args, **kwargs):
            request_data = flask_request.json or flask_request.form.to_dict()

            for param in forbidden_params:
                if param in request_data:
                    return ResponseBuilder.build_error_response(f"Parameter '{param}' is not allowed", settings.RetCode.ARGUMENT_ERROR)

            return function(*args, **kwargs)

        return decorated_function

    return decorator


# Utility Functions
def is_localhost_address(ip_address: str) -> bool:
    """
    Check if an IP address is localhost.

    Args:
        ip_address: IP address to check

    Returns:
        True if localhost, False otherwise
    """
    return ip_address in {"127.0.0.1", "::1", "[::1]", "localhost"}


def handle_server_error(error: Exception) -> Response:
    """
    Handle server errors with proper logging and response formatting.

    Args:
        error: The exception that occurred

    Returns:
        Error response
    """
    logger.exception(f"Server error occurred: {error}")

    try:
        if hasattr(error, "code") and error.code == 401:
            return ResponseBuilder.build_error_response(str(error), settings.RetCode.UNAUTHORIZED)
    except AttributeError:
        pass

    if len(error.args) > 1:
        return ResponseBuilder.build_error_response(str(error.args[0]), settings.RetCode.EXCEPTION_ERROR)

    error_message = str(error)
    if "index_not_found_exception" in error_message:
        return ResponseBuilder.build_error_response("No chunks found. Please upload and parse files first.", settings.RetCode.EXCEPTION_ERROR)

    return ResponseBuilder.build_error_response(error_message, settings.RetCode.EXCEPTION_ERROR)


# Legacy function aliases for backward compatibility
def get_result(code=settings.RetCode.SUCCESS, message="", data=None):
    """Legacy function - use ResponseBuilder.build_success_response instead."""
    return ResponseBuilder.build_success_response(data, message)


def get_error_data_result(message="Sorry! Data missing!", code=settings.RetCode.DATA_ERROR):
    """Legacy function - use ResponseBuilder.build_error_response instead."""
    return ResponseBuilder.build_error_response(message, code)


def get_data_openai(*args, **kwargs):
    """Legacy function - use ResponseBuilder.build_openai_compatible_response instead."""
    return ResponseBuilder.build_openai_compatible_response(*args, **kwargs)


def check_duplicate_ids(ids, id_type="item"):
    """Legacy function - use DataProcessor.check_duplicate_identifiers instead."""
    return DataProcessor.check_duplicate_identifiers(ids, id_type)


def deep_merge(default, custom):
    """Legacy function - use DataProcessor.deep_merge_dictionaries instead."""
    return DataProcessor.deep_merge_dictionaries(default, custom)


def remap_dictionary_keys(source_data, key_aliases=None):
    """Legacy function - use DataProcessor.remap_dictionary_keys instead."""
    return DataProcessor.remap_dictionary_keys(source_data, key_aliases)


def verify_embedding_availability(embd_id, tenant_id):
    """Legacy function - use EmbeddingModelValidator.verify_model_availability instead."""
    return EmbeddingModelValidator.verify_model_availability(embd_id, tenant_id)
