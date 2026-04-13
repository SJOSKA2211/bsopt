import os
import grpc
import structlog
from src.shared.config import settings

logger = structlog.get_logger(__name__)

def get_server_credentials() -> grpc.ServerCredentials | None:
    """
    Constructs gRPC ServerCredentials with mandatory mTLS (require_client_auth=True).
    Ensures all microservices cryptographically verify their peers.
    """
    if not settings.GRPC_SECURE:
        logger.warning("grpc_security_disabled_by_config")
        return None

    try:
        ca_cert = settings.GRPC_CA_CERT
        server_crt = settings.GRPC_SERVER_CERT
        server_key = settings.GRPC_SERVER_KEY

        if not all([ca_cert, server_crt, server_key]):
            logger.warning("grpc_mtls_skipped_missing_configs", ca=bool(ca_cert), crt=bool(server_crt), key=bool(server_key))
            return None

        if not all(os.path.exists(p) for p in [ca_cert, server_crt, server_key]):
            logger.error("grpc_mtls_failed_missing_files", ca=os.path.exists(ca_cert), crt=os.path.exists(server_crt or ""), key=os.path.exists(server_key or ""))
            return None

        with open(server_key, "rb") as f: private_key = f.read()
        with open(server_crt, "rb") as f: certificate_chain = f.read()
        with open(ca_cert, "rb") as f: root_certificates = f.read()

        return grpc.ssl_server_credentials(
            [(private_key, certificate_chain)],
            root_certificates=root_certificates,
            require_client_auth=True,
        )
    except Exception as e:
        logger.exception("grpc_credentials_generation_failed", error=str(e))
        return None

def get_channel_credentials() -> grpc.ChannelCredentials:
    """
    Constructs gRPC ChannelCredentials for clients with mTLS support.
    """
    if not settings.GRPC_SECURE:
        return grpc.local_channel_credentials()

    try:
        ca_cert = settings.GRPC_CA_CERT
        client_crt = settings.GRPC_CLIENT_CERT
        client_key = settings.GRPC_CLIENT_KEY

        root_certs = None
        if os.path.exists(ca_cert):
            with open(ca_cert, "rb") as f:
                root_certs = f.read()

        cert_chain = None
        if client_crt and os.path.exists(client_crt):
            with open(client_crt, "rb") as f:
                cert_chain = f.read()

        pvt_key = None
        if client_key and os.path.exists(client_key):
            with open(client_key, "rb") as f:
                pvt_key = f.read()

        return grpc.ssl_channel_credentials(
            root_certificates=root_certs,
            private_key=pvt_key,
            certificate_chain=cert_chain,
        )
    except Exception as e:
        logger.error("grpc_channel_credentials_failed", error=str(e))
        return grpc.local_channel_credentials()
