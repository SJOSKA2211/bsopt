from typing import Any

from fastapi import HTTPException, status


class BaseAPIException(HTTPException):
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error: str = "InternalServerError"
    message: str = "An unexpected error occurred."
    details: Any | None = None

    def __init__(self, status_code: int | None = None, error: str | None = None, message: str | None = None, details: Any | None = None):
        super().__init__(
            status_code=status_code or self.status_code,
            detail={
                "error": error or self.error,
                "message": message or self.message,
                "details": details or self.details,
            },
        )
        self.status_code = status_code or self.status_code
        self.error = error or self.error
        self.message = message or self.message
        self.details = details or self.details

class NotFoundException(BaseAPIException):
    status_code = status.HTTP_404_NOT_FOUND
    error = "NotFound"
    message = "The requested resource was not found."

class ValidationException(BaseAPIException):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    error = "ValidationError"
    message = "Invalid input provided."

class PermissionDeniedException(BaseAPIException):
    status_code = status.HTTP_403_FORBIDDEN
    error = "PermissionDenied"
    message = "You do not have permission to perform this action."

class AuthenticationException(BaseAPIException):
    status_code = status.HTTP_401_UNAUTHORIZED
    error = "AuthenticationFailed"
    message = "Authentication credentials were not provided or are invalid."

class ConflictException(BaseAPIException):
    status_code = status.HTTP_409_CONFLICT
    error = "Conflict"
    message = "There was a conflict with the current state of the resource."

class ServiceUnavailableException(BaseAPIException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error = "ServiceUnavailable"
    message = "The service is currently unavailable."

class InternalServerException(BaseAPIException):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error = "InternalServerError"
    message = "An unexpected server error occurred."
