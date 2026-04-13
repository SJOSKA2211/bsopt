"""
Custom exceptions for the Auth Service.
These are designed to be translatable to both HTTP and gRPC status codes.
"""

class AuthError(Exception):
    """Base exception for all auth-related errors."""
    def __init__(self, message: str, status_code: int = 401, grpc_code: int = 16): # 16 = Unauthenticated
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.grpc_code = grpc_code

class TokenExpiredError(AuthError):
    """Raised when a JWT token has expired."""
    def __init__(self, message: str = "Token has expired"):
        super().__init__(message, status_code=401, grpc_code=16)

class InvalidTokenError(AuthError):
    """Raised when a JWT token is invalid or signature check fails."""
    def __init__(self, message: str = "Invalid token"):
        super().__init__(message, status_code=401, grpc_code=16)

class TokenRevokedError(AuthError):
    """Raised when a JWT token has been explicitly revoked."""
    def __init__(self, message: str = "Token revoked"):
        super().__init__(message, status_code=401, grpc_code=16)

class InsufficientPermissionsError(AuthError):
    """Raised when a user doesn't have the required role or scope."""
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message, status_code=403, grpc_code=7) # 7 = PermissionDenied

class InvalidCredentialsError(AuthError):
    """Raised when username/password or API key is incorrect."""
    def __init__(self, message: str = "Invalid credentials"):
        super().__init__(message, status_code=401, grpc_code=16)