from typing import Any

from fastapi import status


class BaseAPIException(Exception):
    """Base class for all API exceptions."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "InternalServerError"
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        error_code: str | None = None,
        status_code: int | None = None,
        details: Any | None = None,
    ):
        # OPTIMIZED: Explicit None checks
        if message is not None:
            self.message = message
        if error_code is not None:
            self.error_code = error_code
        if status_code is not None:
            self.status_code = status_code
        self.details = details
        super().__init__(self.message)


class RiskVetoException(BaseAPIException):
    """Thrown when pre-trade risk checks (Solenya Shield) fail."""

    status_code = status.HTTP_403_FORBIDDEN
    error_code = "RiskVeto"
    message = "Order rejected by pre-trade risk engine."


class ExposureLimitException(BaseAPIException):
    """Thrown when portfolio exposure limits are breached."""

    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "ExposureLimitBreach"
    message = "Request would exceed portfolio exposure limits."


class NotFoundException(BaseAPIException):
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "NotFound"
    message = "The requested resource was not found."


class ValidationException(BaseAPIException):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code = "ValidationError"
    message = "The request data is invalid."


class PermissionDeniedException(BaseAPIException):
    status_code = status.HTTP_403_FORBIDDEN
    error_code = "PermissionDenied"
    message = "You do not have permission to perform this action."


class AuthenticationException(BaseAPIException):
    status_code = status.HTTP_401_UNAUTHORIZED
    error_code = "AuthenticationFailed"
    message = "Authentication is required to access this resource."


class ConflictException(BaseAPIException):
    status_code = status.HTTP_409_CONFLICT
    error_code = "Conflict"
    message = "The request conflicts with the current state of the resource."


class ServiceUnavailableException(BaseAPIException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "ServiceUnavailable"
    message = "The service is currently unavailable. Please try again later."


class InternalServerException(BaseAPIException):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "InternalServerError"
    message = "An unexpected internal error occurred."
