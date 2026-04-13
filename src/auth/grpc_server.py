import asyncio
import hashlib
import logging
import os
from datetime import datetime

import grpc
import structlog
from google.protobuf import empty_pb2, timestamp_pb2
from google.protobuf.json_format import MessageToDict, ParseDict
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from src.auth.auth import auth_service
from src.auth.exceptions import AuthError
from src.common.caching import centralized_cache_service
from src.database import db_manager
from src.database.models import APIKey, User
from src.shared.protos import auth_pb2, auth_pb2_grpc
from src.shared.grpc_errors import handle_grpc_error

logger = structlog.get_logger(__name__)


class AuthServicer(auth_pb2_grpc.AuthServiceServicer):
    """Refactored High-Performance Auth gRPC Servicer (BSOPT-v2)."""

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
                roles=token_data.scopes,
            )
        except Exception as e:
            logger.warning("token_validation_failed", error=str(e))
            handle_grpc_error(e, context)
            return auth_pb2.TokenResponse(valid=False)

    async def RefreshToken(self, request, context):
        try:
            token_data = await auth_service.validate_token(request.refresh_token)
            if token_data.token_type != "refresh":
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("invalid_token_type")
                return auth_pb2.TokenResponse(valid=False)

            pair = auth_service.create_token_pair(
                user_id=token_data.user_id, email=token_data.email,
                tier=token_data.tier, scopes=token_data.scopes
            )
            new_access = auth_service.decode_token(pair.access_token)

            return auth_pb2.TokenResponse(
                valid=True, user_id=new_access.user_id, email=new_access.email,
                tier=new_access.tier, expires_at=int(new_access.exp.timestamp()),
                issued_at=int(new_access.iat.timestamp()), token_type="access",
                roles=new_access.scopes
            )
        except Exception as e:
            logger.error("token_refresh_failed", error=str(e))
            handle_grpc_error(e, context)
            return auth_pb2.TokenResponse(valid=False)

    # Simplified user info and key validation using early returns and unified handlers
    async def GetUserInfo(self, request, context):
        try:
            token = await auth_service.validate_token(request.token)
            cached = await centralized_cache_service.get_user_cached(token.user_id)
            if cached:
                return ParseDict(cached, auth_pb2.UserInfo())

            async with db_manager.async_session_factory() as db:
                user = await db.get(User, token.user_id)
                if not user:
                    context.set_code(grpc.StatusCode.NOT_FOUND)
                    return auth_pb2.UserInfo()

                info = auth_pb2.UserInfo(
                    user_id=str(user.id), email=user.email, tier=user.tier,
                    full_name=user.full_name or "", is_verified=user.is_verified,
                    mfa_enabled=user.mfa_enabled, roles=[user.tier]
                )
                info.created_at.FromDatetime(user.created_at)
                if user.last_login_at: info.last_login_at.FromDatetime(user.last_login_at)

                await centralized_cache_service.set_user_cached(token.user_id, MessageToDict(info))
                return info
        except Exception as e:
            handle_grpc_error(e, context)
            return auth_pb2.UserInfo()

    async def ValidateAPIKey(self, request, context):
        try:
            key_hash = hashlib.sha256(request.api_key.encode()).hexdigest()
            cached = await centralized_cache_service.get_api_key_cached(key_hash)
            if cached:
                await centralized_cache_service.update_api_key_last_used(key_hash)
                return ParseDict(cached, auth_pb2.APIKeyResponse())

            async with db_manager.async_session_factory() as db:
                res = await db.execute(select(APIKey).options(joinedload(APIKey.user)).where(APIKey.key_hash == key_hash, APIKey.is_active))
                record = res.scalar_one_or_none()
                if not record: return auth_pb2.APIKeyResponse(valid=False)

                resp = auth_pb2.APIKeyResponse(
                    valid=True, user_id=str(record.user.id), email=record.user.email,
                    tier=record.user.tier, key_name=record.name or ""
                )
                resp.created_at.FromDatetime(record.created_at)
                await centralized_cache_service.set_api_key_cached(key_hash, MessageToDict(resp))
                await centralized_cache_service.update_api_key_last_used(key_hash)
                return resp
        except Exception as e:
            handle_grpc_error(e, context)
            return auth_pb2.APIKeyResponse(valid=False)

    async def IntrospectToken(self, request, context):
        try:
            t = await auth_service.validate_token(request.token)
            return auth_pb2.IntrospectionResponse(
                active=True, sub=t.user_id, username=t.email, token_type=t.token_type,
                exp=int(t.exp.timestamp()), iat=int(t.iat.timestamp()),
                scope=" ".join(t.scopes), iss="bsopt-auth-v2"
            )
        except Exception: return auth_pb2.IntrospectionResponse(active=False)


async def serve(port: str = "50051"):
    options = [
        ("grpc.keepalive_time_ms", 30000),
        ("grpc.keepalive_timeout_ms", 10000),
        ("grpc.keepalive_permit_without_calls", True),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),
        ("grpc.max_connection_idle_ms", 600000),
    ]

    server = grpc.aio.server(options=options)
    auth_pb2_grpc.add_AuthServiceServicer_to_server(AuthServicer(), server)

    health_servicer = health.HealthServicer()
    health_servicer.set("auth.AuthService", health_pb2.HealthCheckResponse.SERVING)
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    listen_addr = f"0.0.0.0:{port}"
    secure_mode = os.getenv("GRPC_SECURE", "true").lower() == "true"
    
    try:
        ca_cert = os.getenv("GRPC_CA_CERT", "/etc/pki/root_ca.crt")
        server_crt = os.getenv("GRPC_SERVER_CERT", "/etc/pki/auth-service.crt")
        server_key = os.getenv("GRPC_SERVER_KEY", "/etc/pki/auth-service.key")

        if secure_mode and all(os.path.exists(p) for p in [ca_cert, server_crt, server_key]):
            with open(server_key, "rb") as f: private_key = f.read()
            with open(server_crt, "rb") as f: certificate_chain = f.read()
            with open(ca_cert, "rb") as f: root_certificates = f.read()

            server_credentials = grpc.ssl_server_credentials(
                [(private_key, certificate_chain)],
                root_certificates=root_certificates,
                require_client_auth=True,
            )
            server.add_secure_port(listen_addr, server_credentials)
            logger.info("grpc_auth_server_online_secure", addr=listen_addr)
        else:
            server.add_insecure_port(listen_addr)
            logger.warning("grpc_auth_server_online_insecure", addr=listen_addr)
    except Exception as e:
        logger.exception("grpc_bootstrap_failed", error=str(e))
        server.add_insecure_port(listen_addr)

    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())