import asyncio

import grpc

from src.protos import auth_pb2, auth_pb2_grpc
from src.security.auth import auth_service


class AuthServicer(auth_pb2_grpc.AuthServiceServicer):
    async def ValidateToken(self, request, context):
        try:
            token_data = auth_service.decode_token(request.token)
            return auth_pb2.TokenResponse(
                valid=True,
                token=request.token,
                role=token_data.tier
            )
        except Exception:
            return auth_pb2.TokenResponse(valid=False)

    async def GenerateToken(self, request, context):
        token_pair = auth_service.create_token_pair(
            user_id=request.user_id,
            email="",  # Optional or fetch from DB
            tier=request.role
        )
        return auth_pb2.TokenResponse(
            valid=True,
            token=token_pair.access_token,
            role=request.role
        )

async def serve():
    server = grpc.aio.server()
    auth_pb2_grpc.add_AuthServiceServicer_to_server(AuthServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("gRPC Auth Server starting on port 50051...")
    await server.start()
    await server.wait_for_termination()

if __name__ == '__main__':
    asyncio.run(serve())
