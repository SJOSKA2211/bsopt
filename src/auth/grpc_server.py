import asyncio

import grpc

from src.auth.auth import auth_service
from src.protos import auth_pb2, auth_pb2_grpc


class AuthServicer(auth_pb2_grpc.AuthServiceServicer):
    async def ValidateToken(self, request, context):
        try:
            token_data = await auth_service.validate_token(request.token)
            return auth_pb2.TokenResponse(
                valid=True,
                token=request.token,
                role=token_data.tier,
                user_id=token_data.user_id,
                email=token_data.email
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            context.set_details(str(e))
            return auth_pb2.TokenResponse(valid=False)

    async def GenerateToken(self, request, context):
        try:
            token_pair = auth_service.create_token_pair(
                user_id=request.user_id,
                email=request.email or "",
                tier=request.role or "free"
            )
            return auth_pb2.TokenResponse(
                valid=True,
                token=token_pair.access_token,
                role=request.role,
                user_id=request.user_id
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return auth_pb2.TokenResponse(valid=False)

async def serve():
    server = grpc.aio.server()
    auth_pb2_grpc.add_AuthServiceServicer_to_server(AuthServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("gRPC Auth Server starting on port 50051...")
    await server.start()
    await server.wait_for_termination()

if __name__ == '__main__':
    asyncio.run(serve())
