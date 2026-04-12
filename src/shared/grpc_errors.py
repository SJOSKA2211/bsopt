import grpc
from src.auth.exceptions import AuthError

def handle_grpc_error(e: Exception, context: grpc.ServicerContext):
    """
    Standardized gRPC error handler that translates custom AuthErrors 
    to appropriate gRPC status codes.
    """
    if isinstance(e, AuthError):
        context.set_code(e.grpc_code)
        context.set_details(str(e))
    else:
        context.set_code(grpc.StatusCode.INTERNAL)
        context.set_details("Internal server error")
