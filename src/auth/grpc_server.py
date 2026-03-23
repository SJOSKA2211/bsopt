import asyncio
import logging
from datetime import UTC, datetime

import grpc
from google.protobuf import timestamp_pb2

from src.auth.auth import auth_service
from src.protos import auth_pb2, auth_pb2_grpc

logger = logging.getLogger(__name__)

class AuthServicer(auth_pb2_grpc.AuthServiceServicer):
    """
    gRPC implementation of the AuthService.
    Bridging zero-trust logic with internal microservices.
    """

    async def ValidateToken(self, request, context):
        try:
            token_data = await auth_service.validate_token(request.token)
            return auth_pb2.TokenResponse(
                valid=True,
                user_id=token_data.user_id,
                email=token_data.email,
                tier=token_data.tier,
                expires_at=int(token_data.exp.timestamp()),
                issued_at=int(token_data.iat.timestamp()),
                token_type=token_data.token_type,
                roles=token_data.scopes
            )
        except Exception as e:
            logger.warning("grpc_validate_token_failed", error=str(e))
            return auth_pb2.TokenResponse(valid=False)

    async def CreateTokenPair(self, request, context):
        try:
            token_pair = auth_service.create_token_pair(
                user_id=request.user_id,
                email=request.email,
                tier=request.tier,
                scopes=[] # Default to empty for internal creation unless specified
            )
            
            issued_at = timestamp_pb2.Timestamp()
            issued_at.GetCurrentTime()

            return auth_pb2.TokenPairResponse(
                access_token=token_pair.access_token,
                refresh_token=token_pair.refresh_token,
                token_type=token_pair.token_type,
                expires_in=token_pair.expires_in,
                issued_at=issued_at
            )
        except Exception as e:
            logger.error("grpc_create_token_pair_failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return auth_pb2.TokenPairResponse()

    async def RevokeToken(self, request, context):
        try:
            await auth_service.revoke_token(request.token)
            from google.protobuf import empty_pb2
            return empty_pb2.Empty()
        except Exception as e:
            logger.error("grpc_revoke_token_failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return auth_pb2.auth__pb2.google_dot_protobuf_dot_empty__pb2.Empty()

async def serve(port: str = "50051"):
    server = grpc.aio.server()
    auth_pb2_grpc.add_AuthServiceServicer_to_server(AuthServicer(), server)
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    logger.info("grpc_auth_server_starting", port=port)
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())
