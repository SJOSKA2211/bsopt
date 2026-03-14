import grpc
import structlog
from src.protos import auth_pb2, auth_pb2_grpc
from src.config import settings

logger = structlog.get_logger(__name__)

class AuthGrpcClient:
    def __init__(self):
        self.channel = None
        self.stub = None
        self.target = settings.AUTH_SERVICE_GRPC_URL # e.g., "auth-service:50051"

    def _get_stub(self):
        if not self.stub:
            self.channel = grpc.aio.insecure_channel(self.target)
            self.stub = auth_pb2_grpc.AuthServiceStub(self.channel)
        return self.stub

    async def validate_token(self, token: str):
        stub = self._get_stub()
        try:
            request = auth_pb2.TokenRequest(token=token)
            response = await stub.ValidateToken(request)
            return response
        except grpc.RpcError as e:
            logger.error("auth_grpc_call_failed", error=str(e))
            return None

    async def close(self):
        if self.channel:
            await self.channel.close()

auth_grpc_client = AuthGrpcClient()
