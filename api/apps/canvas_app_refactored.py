"""
Canvas Application Management API Module

This module handles canvas (agent workflow) management operations including creation,
execution, debugging, and version control. It provides comprehensive canvas lifecycle
management with proper error handling, logging, and authentication.

Key Features:
- Canvas template management
- Canvas lifecycle operations (CRUD)
- Canvas execution with streaming support
- Canvas debugging and testing
- Version control and history management
- Database connectivity testing
- Team collaboration features

Author: RAGFlow Team
License: Apache License 2.0
"""

import json
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from flask import Response, request
from flask_login import current_user, login_required
from peewee import MySQLDatabase, PostgresqlDatabase

from agent.canvas import Canvas
from api.db.db_models import APIToken
from api.db.services.canvas_service import CanvasTemplateService, UserCanvasService
from api.db.services.user_canvas_version import UserCanvasVersionService
from api.db.services.user_service import TenantService
from api.settings import RetCode
from api.utils import get_uuid
from api.utils.api_utils import get_data_error_result, get_json_result, server_error_response, validate_request

# Configure module-specific logger
logger = logging.getLogger(__name__)


class CanvasError(Exception):
    """Custom exception for canvas-related errors."""

    def __init__(self, message: str, code: int = 500):
        self.message = message
        self.code = code
        super().__init__(self.message)


class CanvasAuthorizationManager:
    """Manages canvas authorization and ownership validation."""

    @staticmethod
    def validate_canvas_ownership(user_id: str, canvas_id: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that a user owns a specific canvas.

        Args:
            user_id: User identifier
            canvas_id: Canvas identifier

        Returns:
            Tuple of (is_owner, error_message)
        """
        logger.info(f"Validating canvas ownership for user {user_id}, canvas {canvas_id}")

        if not UserCanvasService.query(user_id=user_id, id=canvas_id):
            error_msg = "Only owner of canvas authorized for this operation."
            logger.warning(f"Canvas ownership validation failed: {error_msg}")
            return False, error_msg

        logger.info("Canvas ownership validation successful")
        return True, None

    @staticmethod
    def validate_api_token_access(canvas_id: str, api_token: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate API token access to a canvas.

        Args:
            canvas_id: Canvas identifier
            api_token: API token for authentication

        Returns:
            Tuple of (is_valid, tenant_id, error_message)
        """
        logger.info(f"Validating API token access for canvas {canvas_id}")

        api_token_objects = APIToken.query(beta=api_token)
        if not api_token_objects:
            error_msg = "Authentication error: API key is invalid!"
            logger.warning(f"API token validation failed: {error_msg}")
            return False, None, error_msg

        tenant_id = api_token_objects[0].tenant_id

        canvas_exists, canvas = UserCanvasService.get_by_id(canvas_id)
        if not canvas_exists or canvas.user_id != tenant_id:
            error_msg = "Canvas not found."
            logger.warning(f"Canvas access validation failed: {error_msg}")
            return False, None, error_msg

        logger.info("API token access validation successful")
        return True, tenant_id, None


class CanvasTemplateManager:
    """Manages canvas template operations."""

    @staticmethod
    def get_all_templates() -> List[Dict[str, Any]]:
        """
        Get all available canvas templates.

        Returns:
            List of template dictionaries
        """
        logger.info("Retrieving all canvas templates")

        try:
            templates = [template.to_dict() for template in CanvasTemplateService.get_all()]
            logger.info(f"Successfully retrieved {len(templates)} canvas templates")
            return templates
        except Exception as error:
            logger.exception(f"Error retrieving canvas templates: {error}")
            return []


class CanvasManager:
    """Manages canvas operations and lifecycle."""

    @staticmethod
    def get_user_canvas_list(user_id: str) -> List[Dict[str, Any]]:
        """
        Get list of canvases for a user, sorted by update time.

        Args:
            user_id: User identifier

        Returns:
            List of canvas dictionaries
        """
        logger.info(f"Retrieving canvas list for user {user_id}")

        try:
            canvas_list = [canvas.to_dict() for canvas in UserCanvasService.query(user_id=user_id)]

            # Sort by update time (newest first)
            sorted_canvas_list = sorted(canvas_list, key=lambda canvas: canvas["update_time"], reverse=True)

            logger.info(f"Successfully retrieved {len(sorted_canvas_list)} canvases for user")
            return sorted_canvas_list

        except Exception as error:
            logger.exception(f"Error retrieving canvas list: {error}")
            return []

    @staticmethod
    def delete_canvases(user_id: str, canvas_ids: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Delete multiple canvases.

        Args:
            user_id: User identifier
            canvas_ids: List of canvas identifiers to delete

        Returns:
            Tuple of (success, error_message)
        """
        logger.info(f"Deleting {len(canvas_ids)} canvases for user {user_id}")

        try:
            for canvas_id in canvas_ids:
                # Validate ownership
                is_owner, error_msg = CanvasAuthorizationManager.validate_canvas_ownership(user_id, canvas_id)
                if not is_owner:
                    return False, error_msg

                # Delete canvas
                UserCanvasService.delete_by_id(canvas_id)

            logger.info(f"Successfully deleted {len(canvas_ids)} canvases")
            return True, None

        except Exception as error:
            logger.exception(f"Error deleting canvases: {error}")
            return False, f"Internal error: {str(error)}"

    @staticmethod
    def save_canvas(user_id: str, canvas_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Save or update a canvas.

        Args:
            user_id: User identifier
            canvas_data: Canvas data to save

        Returns:
            Tuple of (success, result_data or error_info)
        """
        logger.info(f"Saving canvas for user {user_id}")

        try:
            canvas_data["user_id"] = user_id

            # Ensure DSL is properly formatted
            if not isinstance(canvas_data["dsl"], str):
                canvas_data["dsl"] = json.dumps(canvas_data["dsl"], ensure_ascii=False)

            canvas_data["dsl"] = json.loads(canvas_data["dsl"])

            if "id" not in canvas_data:
                # Create new canvas
                return CanvasManager._create_new_canvas(user_id, canvas_data)
            else:
                # Update existing canvas
                return CanvasManager._update_existing_canvas(user_id, canvas_data)

        except Exception as error:
            logger.exception(f"Error saving canvas: {error}")
            return False, {"message": f"Internal error: {str(error)}"}

    @staticmethod
    def _create_new_canvas(user_id: str, canvas_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Create a new canvas."""
        # Check for duplicate title
        if UserCanvasService.query(user_id=user_id, title=canvas_data["title"].strip()):
            return False, {"message": f"{canvas_data['title'].strip()} already exists."}

        canvas_data["id"] = get_uuid()

        if not UserCanvasService.save(**canvas_data):
            return False, {"message": "Failed to save canvas."}

        # Save version
        CanvasManager._save_canvas_version(canvas_data)

        logger.info(f"Successfully created new canvas {canvas_data['id']}")
        return True, canvas_data

    @staticmethod
    def _update_existing_canvas(user_id: str, canvas_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Update an existing canvas."""
        # Validate ownership
        is_owner, error_msg = CanvasAuthorizationManager.validate_canvas_ownership(user_id, canvas_data["id"])
        if not is_owner:
            return False, {"message": error_msg}

        UserCanvasService.update_by_id(canvas_data["id"], canvas_data)

        # Save version
        CanvasManager._save_canvas_version(canvas_data)

        logger.info(f"Successfully updated canvas {canvas_data['id']}")
        return True, canvas_data

    @staticmethod
    def _save_canvas_version(canvas_data: Dict[str, Any]) -> None:
        """Save a version of the canvas."""
        version_title = f"{canvas_data['title']}_{time.strftime('%Y_%m_%d_%H_%M_%S')}"

        UserCanvasVersionService.insert(user_canvas_id=canvas_data["id"], dsl=canvas_data["dsl"], title=version_title)

        # Clean up old versions
        UserCanvasVersionService.delete_all_versions(canvas_data["id"])

    @staticmethod
    def get_canvas_by_id(canvas_id: str, user_id: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Get a canvas by ID with optional user validation.

        Args:
            canvas_id: Canvas identifier
            user_id: Optional user identifier for ownership validation

        Returns:
            Tuple of (success, canvas_data or error_info)
        """
        logger.info(f"Retrieving canvas {canvas_id}")

        try:
            canvas_exists, canvas = UserCanvasService.get_by_tenant_id(canvas_id)

            if not canvas_exists:
                return False, {"message": "Canvas not found."}

            if user_id and canvas["user_id"] != user_id:
                return False, {"message": "Canvas not found."}

            logger.info(f"Successfully retrieved canvas {canvas_id}")
            return True, canvas

        except Exception as error:
            logger.exception(f"Error retrieving canvas {canvas_id}: {error}")
            return False, {"message": f"Internal error: {str(error)}"}


class CanvasExecutionManager:
    """Manages canvas execution operations."""

    @staticmethod
    def execute_canvas(user_id: str, canvas_id: str, execution_params: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Execute a canvas with given parameters.

        Args:
            user_id: User identifier
            canvas_id: Canvas identifier
            execution_params: Execution parameters

        Returns:
            Tuple of (success, response or error_info)
        """
        logger.info(f"Executing canvas {canvas_id} for user {user_id}")

        try:
            # Get canvas
            canvas_exists, canvas_service = UserCanvasService.get_by_id(canvas_id)
            if not canvas_exists:
                return False, {"message": "Canvas not found."}

            # Validate ownership
            is_owner, error_msg = CanvasAuthorizationManager.validate_canvas_ownership(user_id, canvas_id)
            if not is_owner:
                return False, {"message": error_msg}

            # Prepare canvas DSL
            if not isinstance(canvas_service.dsl, str):
                canvas_service.dsl = json.dumps(canvas_service.dsl, ensure_ascii=False)

            # Initialize canvas
            canvas = Canvas(canvas_service.dsl, user_id)

            # Add user message if provided
            message_id = execution_params.get("message_id", get_uuid())
            if "message" in execution_params:
                canvas.messages.append({"role": "user", "content": execution_params["message"], "id": message_id})
                canvas.add_user_input(execution_params["message"])

            # Execute canvas
            is_streaming = execution_params.get("stream", True)
            running_hint_text = execution_params.get("running_hint_text", "")

            if is_streaming:
                return CanvasExecutionManager._execute_streaming(canvas, canvas_service, canvas_id, message_id, running_hint_text)
            else:
                return CanvasExecutionManager._execute_non_streaming(canvas, canvas_service, canvas_id, message_id, running_hint_text)

        except Exception as error:
            logger.exception(f"Error executing canvas {canvas_id}: {error}")
            return False, {"message": f"Internal error: {str(error)}"}

    @staticmethod
    def _execute_streaming(canvas: Canvas, canvas_service: Any, canvas_id: str, message_id: str, running_hint_text: str) -> Tuple[bool, Response]:
        """Execute canvas with streaming response."""

        def generate_streaming_response():
            """Generate streaming response chunks."""
            final_answer = {"reference": [], "content": ""}

            try:
                for answer_chunk in canvas.run(running_hint_text=running_hint_text, stream=True):
                    if answer_chunk.get("running_status"):
                        yield "data:" + json.dumps({"code": 0, "message": "", "data": {"answer": answer_chunk["content"], "running_status": True}}, ensure_ascii=False) + "\n\n"
                        continue

                    # Update final answer
                    for key in answer_chunk.keys():
                        final_answer[key] = answer_chunk[key]

                    response_data = {"answer": answer_chunk["content"], "reference": answer_chunk.get("reference", [])}

                    yield "data:" + json.dumps({"code": 0, "message": "", "data": response_data}, ensure_ascii=False) + "\n\n"

                # Update canvas state
                CanvasExecutionManager._update_canvas_after_execution(canvas, canvas_service, canvas_id, message_id, final_answer)

            except Exception as error:
                logger.exception(f"Error in streaming execution: {error}")

                # Update canvas state even on error
                canvas_service.dsl = json.loads(str(canvas))
                if not canvas.path[-1]:
                    canvas.path.pop(-1)
                UserCanvasService.update_by_id(canvas_id, canvas_service.to_dict())

                yield "data:" + json.dumps({"code": 500, "message": str(error), "data": {"answer": f"**ERROR**: {str(error)}", "reference": []}}, ensure_ascii=False) + "\n\n"

            yield "data:" + json.dumps({"code": 0, "message": "", "data": True}, ensure_ascii=False) + "\n\n"

        response = Response(generate_streaming_response(), mimetype="text/event-stream")
        response.headers.add_header("Cache-control", "no-cache")
        response.headers.add_header("Connection", "keep-alive")
        response.headers.add_header("X-Accel-Buffering", "no")
        response.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")

        return True, response

    @staticmethod
    def _execute_non_streaming(canvas: Canvas, canvas_service: Any, canvas_id: str, message_id: str, running_hint_text: str) -> Tuple[bool, Dict[str, Any]]:
        """Execute canvas with non-streaming response."""
        final_answer = {"content": "", "reference": []}

        for answer_chunk in canvas.run(running_hint_text=running_hint_text, stream=False):
            if answer_chunk.get("running_status"):
                continue

            final_answer["content"] = "\n".join(answer_chunk["content"]) if "content" in answer_chunk else ""

            if final_answer.get("reference"):
                canvas.reference.append(final_answer["reference"])

            # Update canvas state
            CanvasExecutionManager._update_canvas_after_execution(canvas, canvas_service, canvas_id, message_id, final_answer)

            return True, {"answer": final_answer["content"], "reference": final_answer.get("reference", [])}

        return False, {"message": "Canvas execution failed"}

    @staticmethod
    def _update_canvas_after_execution(canvas: Canvas, canvas_service: Any, canvas_id: str, message_id: str, final_answer: Dict[str, Any]) -> None:
        """Update canvas state after execution."""
        canvas.messages.append({"role": "assistant", "content": final_answer["content"], "id": message_id})
        canvas.history.append(("assistant", final_answer["content"]))

        if not canvas.path[-1]:
            canvas.path.pop(-1)

        if final_answer.get("reference"):
            canvas.reference.append(final_answer["reference"])

        canvas_service.dsl = json.loads(str(canvas))
        UserCanvasService.update_by_id(canvas_id, canvas_service.to_dict())


class CanvasResetManager:
    """Manages canvas reset operations."""

    @staticmethod
    def reset_canvas(user_id: str, canvas_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Reset a canvas to its initial state.

        Args:
            user_id: User identifier
            canvas_id: Canvas identifier

        Returns:
            Tuple of (success, reset_dsl or error_info)
        """
        logger.info(f"Resetting canvas {canvas_id} for user {user_id}")

        try:
            # Get canvas
            canvas_exists, user_canvas = UserCanvasService.get_by_id(canvas_id)
            if not canvas_exists:
                return False, {"message": "Canvas not found."}

            # Validate ownership
            is_owner, error_msg = CanvasAuthorizationManager.validate_canvas_ownership(user_id, canvas_id)
            if not is_owner:
                return False, {"message": error_msg}

            # Reset canvas
            canvas = Canvas(json.dumps(user_canvas.dsl), user_id)
            canvas.reset()

            reset_dsl = json.loads(str(canvas))
            UserCanvasService.update_by_id(canvas_id, {"dsl": reset_dsl})

            logger.info(f"Successfully reset canvas {canvas_id}")
            return True, reset_dsl

        except Exception as error:
            logger.exception(f"Error resetting canvas {canvas_id}: {error}")
            return False, {"message": f"Internal error: {str(error)}"}


class CanvasDebugManager:
    """Manages canvas debugging operations."""

    @staticmethod
    def get_input_elements(user_id: str, canvas_id: str, component_id: str) -> Tuple[bool, Any]:
        """
        Get input elements for a canvas component.

        Args:
            user_id: User identifier
            canvas_id: Canvas identifier
            component_id: Component identifier

        Returns:
            Tuple of (success, input_elements or error_info)
        """
        logger.info(f"Getting input elements for canvas {canvas_id}, component {component_id}")

        try:
            # Get canvas
            canvas_exists, user_canvas = UserCanvasService.get_by_id(canvas_id)
            if not canvas_exists:
                return False, {"message": "Canvas not found."}

            # Validate ownership
            is_owner, error_msg = CanvasAuthorizationManager.validate_canvas_ownership(user_id, canvas_id)
            if not is_owner:
                return False, {"message": error_msg}

            # Get input elements
            canvas = Canvas(json.dumps(user_canvas.dsl), user_id)
            input_elements = canvas.get_component_input_elements(component_id)

            logger.info(f"Successfully retrieved input elements for component {component_id}")
            return True, input_elements

        except Exception as error:
            logger.exception(f"Error getting input elements: {error}")
            return False, {"message": f"Internal error: {str(error)}"}

    @staticmethod
    def debug_component(user_id: str, canvas_id: str, component_id: str, debug_params: List[Dict[str, Any]]) -> Tuple[bool, Any]:
        """
        Debug a canvas component with given parameters.

        Args:
            user_id: User identifier
            canvas_id: Canvas identifier
            component_id: Component identifier
            debug_params: Debug parameters

        Returns:
            Tuple of (success, debug_result or error_info)
        """
        logger.info(f"Debugging canvas {canvas_id}, component {component_id}")

        try:
            # Validate parameters
            for param in debug_params:
                if not param.get("key"):
                    return False, {"message": "Parameter key is required"}

            # Get canvas
            canvas_exists, user_canvas = UserCanvasService.get_by_id(canvas_id)
            if not canvas_exists:
                return False, {"message": "Canvas not found."}

            # Validate ownership
            is_owner, error_msg = CanvasAuthorizationManager.validate_canvas_ownership(user_id, canvas_id)
            if not is_owner:
                return False, {"message": error_msg}

            # Debug component
            canvas = Canvas(json.dumps(user_canvas.dsl), user_id)
            component = canvas.get_component(component_id)["obj"]
            component.reset()
            component._param.debug_inputs = debug_params

            debug_result = component.debug()
            result_data = debug_result.to_dict(orient="records")

            logger.info(f"Successfully debugged component {component_id}")
            return True, result_data

        except Exception as error:
            logger.exception(f"Error debugging component: {error}")
            return False, {"message": f"Internal error: {str(error)}"}


class DatabaseConnectionManager:
    """Manages database connection testing."""

    @staticmethod
    def test_database_connection(connection_params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Test database connection with given parameters.

        Args:
            connection_params: Database connection parameters

        Returns:
            Tuple of (success, result_message)
        """
        logger.info(f"Testing database connection for {connection_params.get('db_type')}")

        try:
            db_type = connection_params["db_type"]
            database_name = connection_params["database"]
            username = connection_params["username"]
            host = connection_params["host"]
            port = connection_params["port"]
            password = connection_params["password"]

            if db_type in ["mysql", "mariadb"]:
                database = MySQLDatabase(database_name, user=username, host=host, port=port, password=password)
            elif db_type == "postgresql":
                database = PostgresqlDatabase(database_name, user=username, host=host, port=port, password=password)
            else:
                return False, "Unsupported database type."

            # Test connection
            database.connect()
            database.close()

            logger.info(f"Database connection test successful for {db_type}")
            return True, "Database Connection Successful!"

        except Exception as error:
            logger.exception(f"Database connection test failed: {error}")
            return False, str(error)


class CanvasVersionManager:
    """Manages canvas version operations."""

    @staticmethod
    def get_canvas_versions(canvas_id: str) -> List[Dict[str, Any]]:
        """
        Get all versions of a canvas.

        Args:
            canvas_id: Canvas identifier

        Returns:
            List of version dictionaries
        """
        logger.info(f"Retrieving versions for canvas {canvas_id}")

        try:
            versions = [version.to_dict() for version in UserCanvasVersionService.list_by_canvas_id(canvas_id)]

            # Sort by update time (newest first)
            sorted_versions = sorted(versions, key=lambda version: version["update_time"], reverse=True)

            logger.info(f"Successfully retrieved {len(sorted_versions)} versions")
            return sorted_versions

        except Exception as error:
            logger.exception(f"Error retrieving canvas versions: {error}")
            return []

    @staticmethod
    def get_canvas_version(version_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Get a specific canvas version.

        Args:
            version_id: Version identifier

        Returns:
            Tuple of (success, version_data or error_info)
        """
        logger.info(f"Retrieving canvas version {version_id}")

        try:
            version_exists, version = UserCanvasVersionService.get_by_id(version_id)

            if version_exists:
                logger.info(f"Successfully retrieved version {version_id}")
                return True, version.to_dict()
            else:
                return False, {"message": "Version not found"}

        except Exception as error:
            logger.exception(f"Error retrieving canvas version: {error}")
            return False, {"message": f"Internal error: {str(error)}"}


class TeamCanvasManager:
    """Manages team canvas operations."""

    @staticmethod
    def get_team_canvases(user_id: str, query_params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Get team canvases with filtering and pagination.

        Args:
            user_id: User identifier
            query_params: Query parameters

        Returns:
            Tuple of (success, result_data or error_info)
        """
        logger.info(f"Retrieving team canvases for user {user_id}")

        try:
            keywords = query_params.get("keywords", "")
            page_number = int(query_params.get("page", 1))
            items_per_page = int(query_params.get("page_size", 150))
            order_by = query_params.get("orderby", "create_time")
            is_descending = query_params.get("desc", True)

            # Get joined tenants
            tenants = TenantService.get_joined_tenants_by_user_id(user_id)
            tenant_ids = [tenant["tenant_id"] for tenant in tenants]

            # Get canvases
            canvases, total = UserCanvasService.get_by_tenant_ids(tenant_ids, user_id, page_number, items_per_page, order_by, is_descending, keywords)

            result_data = {"kbs": canvases, "total": total}

            logger.info(f"Successfully retrieved {len(canvases)} team canvases")
            return True, result_data

        except Exception as error:
            logger.exception(f"Error retrieving team canvases: {error}")
            return False, {"message": f"Internal error: {str(error)}"}


class CanvasSettingsManager:
    """Manages canvas settings operations."""

    @staticmethod
    def update_canvas_settings(user_id: str, canvas_id: str, settings_data: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Update canvas settings.

        Args:
            user_id: User identifier
            canvas_id: Canvas identifier
            settings_data: Settings data to update

        Returns:
            Tuple of (success, update_count or error_info)
        """
        logger.info(f"Updating settings for canvas {canvas_id}")

        try:
            # Get canvas
            canvas_exists, canvas = UserCanvasService.get_by_id(canvas_id)
            if not canvas_exists:
                return False, {"message": "Canvas not found."}

            # Validate ownership
            is_owner, error_msg = CanvasAuthorizationManager.validate_canvas_ownership(user_id, canvas_id)
            if not is_owner:
                return False, {"message": error_msg}

            # Prepare update data
            canvas_dict = canvas.to_dict()
            canvas_dict["title"] = settings_data["title"]

            if settings_data.get("description"):
                canvas_dict["description"] = settings_data["description"]

            if settings_data.get("permission"):
                canvas_dict["permission"] = settings_data["permission"]

            if settings_data.get("avatar"):
                canvas_dict["avatar"] = settings_data["avatar"]

            # Update canvas
            update_count = UserCanvasService.update_by_id(canvas_id, canvas_dict)

            logger.info(f"Successfully updated settings for canvas {canvas_id}")
            return True, update_count

        except Exception as error:
            logger.exception(f"Error updating canvas settings: {error}")
            return False, {"message": f"Internal error: {str(error)}"}


# Route handlers using the new managers
@manager.route("/templates", methods=["GET"])  # noqa: F821
@login_required
def get_canvas_templates_endpoint():
    """Get all canvas templates."""
    templates = CanvasTemplateManager.get_all_templates()
    return get_json_result(data=templates)


@manager.route("/list", methods=["GET"])  # noqa: F821
@login_required
def get_canvas_list_endpoint():
    """Get user's canvas list."""
    canvas_list = CanvasManager.get_user_canvas_list(current_user.id)
    return get_json_result(data=canvas_list)


@manager.route("/rm", methods=["POST"])  # noqa: F821
@validate_request("canvas_ids")
@login_required
def delete_canvases_endpoint():
    """Delete multiple canvases."""
    canvas_ids = request.json["canvas_ids"]

    success, error_msg = CanvasManager.delete_canvases(current_user.id, canvas_ids)

    if success:
        return get_json_result(data=True)
    else:
        return get_json_result(data=False, message=error_msg, code=RetCode.OPERATING_ERROR)


@manager.route("/set", methods=["POST"])  # noqa: F821
@validate_request("dsl", "title")
@login_required
def save_canvas_endpoint():
    """Save or update a canvas."""
    canvas_data = request.json

    success, result = CanvasManager.save_canvas(current_user.id, canvas_data)

    if success:
        return get_json_result(data=result)
    else:
        return get_data_error_result(message=result["message"])


@manager.route("/get/<canvas_id>", methods=["GET"])  # noqa: F821
@login_required
def get_canvas_endpoint(canvas_id: str):
    """Get a canvas by ID."""
    success, result = CanvasManager.get_canvas_by_id(canvas_id, current_user.id)

    if success:
        return get_json_result(data=result)
    else:
        return get_data_error_result(message=result["message"])


@manager.route("/getsse/<canvas_id>", methods=["GET"])  # noqa: F821
def get_canvas_sse_endpoint(canvas_id: str):
    """Get canvas via SSE with API token authentication."""
    authorization_header = request.headers.get("Authorization", "")
    auth_parts = authorization_header.split()

    if len(auth_parts) != 2:
        return get_data_error_result(message="Authorization is not valid!")

    token = auth_parts[1]
    is_valid, tenant_id, error_msg = CanvasAuthorizationManager.validate_api_token_access(canvas_id, token)

    if not is_valid:
        return get_data_error_result(message=error_msg)

    success, result = CanvasManager.get_canvas_by_id(canvas_id)

    if success:
        return get_json_result(data=result)
    else:
        return get_data_error_result(message=result["message"])


@manager.route("/completion", methods=["POST"])  # noqa: F821
@validate_request("id")
@login_required
def execute_canvas_endpoint():
    """Execute a canvas."""
    execution_params = request.json
    canvas_id = execution_params["id"]

    success, result = CanvasExecutionManager.execute_canvas(current_user.id, canvas_id, execution_params)

    if success:
        return result  # Response object for streaming or dict for non-streaming
    else:
        return server_error_response(Exception(result["message"]))


@manager.route("/reset", methods=["POST"])  # noqa: F821
@validate_request("id")
@login_required
def reset_canvas_endpoint():
    """Reset a canvas to initial state."""
    canvas_id = request.json["id"]

    success, result = CanvasResetManager.reset_canvas(current_user.id, canvas_id)

    if success:
        return get_json_result(data=result)
    else:
        return server_error_response(Exception(result["message"]))


@manager.route("/input_elements", methods=["GET"])  # noqa: F821
@login_required
def get_input_elements_endpoint():
    """Get input elements for a canvas component."""
    canvas_id = request.args.get("id")
    component_id = request.args.get("component_id")

    success, result = CanvasDebugManager.get_input_elements(current_user.id, canvas_id, component_id)

    if success:
        return get_json_result(data=result)
    else:
        return server_error_response(Exception(result["message"]))


@manager.route("/debug", methods=["POST"])  # noqa: F821
@validate_request("id", "component_id", "params")
@login_required
def debug_canvas_component_endpoint():
    """Debug a canvas component."""
    debug_data = request.json
    canvas_id = debug_data["id"]
    component_id = debug_data["component_id"]
    debug_params = debug_data["params"]

    success, result = CanvasDebugManager.debug_component(current_user.id, canvas_id, component_id, debug_params)

    if success:
        return get_json_result(data=result)
    else:
        return server_error_response(Exception(result["message"]))


@manager.route("/test_db_connect", methods=["POST"])  # noqa: F821
@validate_request("db_type", "database", "username", "host", "port", "password")
@login_required
def test_database_connection_endpoint():
    """Test database connection."""
    connection_params = request.json

    success, result_message = DatabaseConnectionManager.test_database_connection(connection_params)

    if success:
        return get_json_result(data=result_message)
    else:
        return server_error_response(Exception(result_message))


@manager.route("/getlistversion/<canvas_id>", methods=["GET"])  # noqa: F821
@login_required
def get_canvas_versions_endpoint(canvas_id: str):
    """Get list of canvas versions."""
    versions = CanvasVersionManager.get_canvas_versions(canvas_id)
    return get_json_result(data=versions)


@manager.route("/getversion/<version_id>", methods=["GET"])  # noqa: F821
@login_required
def get_canvas_version_endpoint(version_id: str):
    """Get a specific canvas version."""
    success, result = CanvasVersionManager.get_canvas_version(version_id)

    if success:
        return get_json_result(data=result)
    else:
        return get_json_result(data=result["message"])


@manager.route("/listteam", methods=["GET"])  # noqa: F821
@login_required
def get_team_canvases_endpoint():
    """Get team canvases with filtering and pagination."""
    query_params = {
        "keywords": request.args.get("keywords", ""),
        "page": request.args.get("page", 1),
        "page_size": request.args.get("page_size", 150),
        "orderby": request.args.get("orderby", "create_time"),
        "desc": request.args.get("desc", True),
    }

    success, result = TeamCanvasManager.get_team_canvases(current_user.id, query_params)

    if success:
        return get_json_result(data=result)
    else:
        return server_error_response(Exception(result["message"]))


@manager.route("/setting", methods=["POST"])  # noqa: F821
@validate_request("id", "title", "permission")
@login_required
def update_canvas_settings_endpoint():
    """Update canvas settings."""
    settings_data = request.json
    canvas_id = settings_data["id"]

    success, result = CanvasSettingsManager.update_canvas_settings(current_user.id, canvas_id, settings_data)

    if success:
        return get_json_result(data=result)
    else:
        return get_data_error_result(message=result["message"])
