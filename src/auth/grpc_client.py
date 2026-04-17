import asyncio

import grpc
import structlog

from src.shared.config import settings
from src.shared.protos import auth_pb2, auth_pb2_grpc

logger = structlog.get_logger(__name__)


class AuthGrpcClient:
    def __init__(self):
        self.channel = None
        self.stub = None
        self.target = settings.AUTH_SERVICE_GRPC_URL  # e.g., "auth-service:50051"

    def _get_stub(self):
        if not self.stub:
            from src.shared.grpc_util import get_channel_credentials

            options = [
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
                ("grpc.keepalive_permit_without_calls", True),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.enable_retries", 1),
            ]

            import json
            service_config = {
                "methodConfig": [{
                    "name": [{"service": "auth.AuthService"}],
                    "retryPolicy": {
                        "maxAttempts": 5,
                        "initialBackoff": "0.1s",
                        "maxBackoff": "1s",
                        "backoffMultiplier": 2,
                        "retryableStatusCodes": ["UNAVAILABLE", "INTERNAL"],
                    },
                }]
            }
            options.append(("grpc.service_config", json.dumps(service_config)))

            try:
                credentials = get_channel_credentials()
                if isinstance(credentials, grpc.ChannelCredentials):
                    self.channel = grpc.aio.secure_channel(self.target, credentials, options=options)
                    logger.info("grpc_client_init_secure", target=self.target)
                else:
                    self.channel = grpc.aio.insecure_channel(self.target, options=options)
                    logger.warning("grpc_client_init_insecure", target=self.target)
            except Exception as e:
                logger.exception("grpc_client_init_failed", error=str(e))
                self.channel = grpc.aio.insecure_channel(self.target, options=options)

            self.stub = auth_pb2_grpc.AuthServiceStub(self.channel)
        return self.stub

    async def validate_token(self, token: str):
        stub = self._get_stub()
        try:
            request = auth_pb2.TokenRequest(token=token)
            # Timeout for RPC calls
            response = await asyncio.wait_for(stub.ValidateToken(request), timeout=5.0)
            return response
        except TimeoutError:
            logger.error("auth_grpc_timeout")
            return None
        except grpc.RpcError as e:
            logger.error("auth_grpc_call_failed", status=e.code(), details=e.details())
            return None

    async def close(self):
        if self.channel:
            await self.channel.close()


auth_grpc_client = AuthGrpcClient()