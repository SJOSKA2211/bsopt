"""
Manifold gRPC Connection Manager (Phase 3).
Implements connection pooling and mTLS lifecycle management
to prevent "Auth dropping" during high-throughput requests.
"""

import logging
import os
from typing import Optional

import grpc

logger = logging.getLogger(__name__)

class GRPCManager:
    _instance: Optional['GRPCManager'] = None
    _auth_channel: grpc.aio.Channel | None = None

    def __init__(self):
        self.auth_addr = os.getenv("AUTH_SVC_ADDR", "auth_service:50051")
        self._ca_cert_path = "/etc/ssl/certs/root_ca.crt"
        self._client_cert_path = "/etc/ssl/certs/api_service.crt"
        self._client_key_path = "/etc/ssl/certs/api_service.key"

    @classmethod
    def get_instance(cls) -> 'GRPCManager':
        if cls._instance is None:
            cls._instance = GRPCManager()
        return cls._instance

    def _get_credentials(self) -> grpc.ChannelCredentials | None:
        """Load mTLS credentials (Axiom: DevSecOps Phase 3)."""
        if all(os.path.exists(p) for p in [self._ca_cert_path, self._client_cert_path, self._client_key_path]):
            with open(self._ca_cert_path, "rb") as f:
                root_certs = f.read()
            with open(self._client_cert_path, "rb") as f:
                cert_chain = f.read()
            with open(self._client_key_path, "rb") as f:
                private_key = f.read()
            
            return grpc.ssl_channel_credentials(
                root_certificates=root_certs,
                private_key=private_key,
                certificate_chain=cert_chain
            )
        return None

    async def get_auth_channel(self) -> grpc.aio.Channel:
        """Retrieve, initialize, or recover the persistent gRPC channel."""
        recreate = False
        if self._auth_channel is not None:
            # Check state of existing channel
            state = self._auth_channel.get_state(True)
            if state in (grpc.ChannelConnectivity.SHUTDOWN, grpc.ChannelConnectivity.TRANSIENT_FAILURE):
                logger.warning("Existing gRPC channel is in terminal state: %s. Re-initializing...", state)
                await self.close()
                recreate = True

        if self._auth_channel is None or recreate:
            creds = self._get_credentials()
            options = [
                ('grpc.keepalive_time_ms', 10000),
                ('grpc.keepalive_timeout_ms', 5000),
                ('grpc.keepalive_permit_without_calls', True),
                ('grpc.http2.max_pings_without_data', 0),
                ('grpc.http2.min_sent_ping_interval_without_data_ms', 5000),
            ]
            if creds:
                logger.info("Initializing PERSISTENT SECURE gRPC channel for %s", self.auth_addr)
                self._auth_channel = grpc.aio.secure_channel(self.auth_addr, creds, options=options)
            else:
                logger.warning("Initializing PERSISTENT INSECURE gRPC channel for %s", self.auth_addr)
                self._auth_channel = grpc.aio.insecure_channel(self.auth_addr, options=options)
        
        return self._auth_channel

    async def close(self):
        """Cleanly shutdown channels."""
        if self._auth_channel:
            await self._auth_channel.close()
            self._auth_channel = None

grpc_manager = GRPCManager.get_instance()
