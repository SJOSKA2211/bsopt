import os
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
            # gRPC options for stability (Keepalive & HTTP/2)
            # Retries enabled via service config
            options = [
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
                ("grpc.keepalive_permit_without_calls", True),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.enable_retries", 1),
            ]

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
            options.append(("grpc.service_config", str(service_config)))

            SECURE_MODE = os.getenv("GRPC_SECURE", "true").lower() == "true"

            try:
                # mTLS Configuration
                PROJECT_ROOT = os.getcwd()
                # Use environment variables or default to standard paths
                ca_cert_path = os.getenv("GRPC_CA_CERT", os.path.join(PROJECT_ROOT, ".pki/root_ca.crt"))
                client_cert_path = os.getenv("GRPC_CLIENT_CERT", os.path.join(PROJECT_ROOT, ".pki/api.crt"))
                client_key_path = os.getenv("GRPC_CLIENT_KEY", os.path.join(PROJECT_ROOT, ".pki/api.key"))

                if SECURE_MODE and os.path.exists(ca_cert_path) and os.path.exists(client_cert_path):
                    with open(ca_cert_path, "rb") as f:
                        root_certs = f.read()
                    with open(client_cert_path, "rb") as f:
                        client_cert = f.read()
                    with open(client_key_path, "rb") as f:
                        client_key = f.read()

                    credentials = grpc.ssl_channel_credentials(
                        root_certificates=root_certs,
                        private_key=client_key,
                        certificate_chain=client_cert,
                    )
                    self.channel = grpc.aio.secure_channel(self.target, credentials, options=options)
                    logger.info("grpc_client_init_secure", target=self.target)
                else:
                    if SECURE_MODE:
                        logger.warning("grpc_certs_not_found_falling_back_to_insecure", target=self.target)
                    self.channel = grpc.aio.insecure_channel(self.target, options=options)
            except Exception as e:
                logger.exception("grpc_client_secure_init_failed", error=str(e))
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
        except asyncio.TimeoutError:
            logger.error("auth_grpc_timeout")
            return None
        except grpc.RpcError as e:
            logger.error("auth_grpc_call_failed", status=e.code(), details=e.details())
            return None

    async def close(self):
        if self.channel:
            await self.channel.close()


auth_grpc_client = AuthGrpcClient()
