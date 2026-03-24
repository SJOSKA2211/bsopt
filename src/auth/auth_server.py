import grpc
import structlog

from src.auth.auth import auth_service
from src.shared.protos import auth_pb2, auth_pb2_grpc

logger = structlog.get_logger(__name__)

class AuthServicer(auth_pb2_grpc.AuthServiceServicer):
    """
    gRPC Servicer for Production-Grade Authentication.
    Uses Argon2id and Asymmetric JWTs.
    """

    async def ValidateToken(self, request, context):
        try:
            token_data = await auth_service.validate_token(request.token)
            return auth_pb2.TokenResponse(valid=True, token=request.token, role=token_data.tier)
        except Exception as e:
            logger.error("grpc_auth_validation_failed", error=str(e))
            return auth_pb2.TokenResponse(valid=False)

    async def GenerateToken(self, request, context):
        # In a real gRPC flow, this might be called after internal verification
        token_pair = auth_service.create_token_pair(request.user_id, "", request.role)
        return auth_pb2.TokenResponse(valid=True, token=token_pair.access_token, role=request.role)

async def serve():
    server = grpc.aio.server()
    auth_pb2_grpc.add_AuthServiceServicer_to_server(AuthServicer(), server)
    listen_addr = "[::]:50051"
    server.add_insecure_port(listen_addr)
    logger.info("auth_grpc_server_started", addr=listen_addr)
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    import asyncio

    asyncio.run(serve())
