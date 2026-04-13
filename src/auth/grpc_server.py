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
    """
    Consolidated High-Performance Auth gRPC Servicer.
    Implements all methods defined in auth.proto for zero-trust internal comms.
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
                roles=token_data.scopes,
            )
        except Exception as e:
            logger.warning("grpc_validate_token_failed", error=str(e))
            handle_grpc_error(e, context)
            return auth_pb2.TokenResponse(valid=False)

    async def RefreshToken(self, request, context):
        try:
            token_data = await auth_service.validate_token(request.refresh_token)
            if token_data.token_type != "refresh":
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Token is not a refresh token")
                return auth_pb2.TokenResponse(valid=False)

            token_pair = auth_service.create_token_pair(
                user_id=token_data.user_id,
                email=token_data.email,
                tier=token_data.tier,
                scopes=token_data.scopes,
            )

            new_access_data = auth_service.decode_token(token_pair.access_token)

            return auth_pb2.TokenResponse(
                valid=True,
                user_id=new_access_data.user_id,
                email=new_access_data.email,
                tier=new_access_data.tier,
                expires_at=int(new_access_data.exp.timestamp()),
                issued_at=int(new_access_data.iat.timestamp()),
                token_type="access",
                roles=new_access_data.scopes,
            )
        except Exception as e:
            logger.error("grpc_refresh_token_failed", error=str(e))
            handle_grpc_error(e, context)
            return auth_pb2.TokenResponse(valid=False)

    async def RevokeToken(self, request, context):
        try:
            await auth_service.revoke_token(request.token)
            return empty_pb2.Empty()
        except Exception as e:
            logger.error("grpc_revoke_token_failed", error=str(e))
            handle_grpc_error(e, context)
            return empty_pb2.Empty()

    async def GetUserInfo(self, request, context):
        try:
            token_data = await auth_service.validate_token(request.token)

            cached_user_data = await centralized_cache_service.get_user_cached(token_data.user_id)
            if cached_user_data:
                user_info = auth_pb2.UserInfo()
                ParseDict(cached_user_data, user_info)
                return user_info

            async with db_manager.async_session_factory() as db:
                result = await db.execute(select(User).where(User.id == token_data.user_id))
                user = result.scalar_one_or_none()
                if not user:
                    context.set_code(grpc.StatusCode.NOT_FOUND)
                    return auth_pb2.UserInfo()

                created_at = timestamp_pb2.Timestamp()
                created_at.FromDatetime(user.created_at)

                last_login = timestamp_pb2.Timestamp()
                if user.last_login_at:
                    last_login.FromDatetime(user.last_login_at)

                user_info = auth_pb2.UserInfo(
                    user_id=str(user.id),
                    email=user.email,
                    tier=user.tier,
                    full_name=user.full_name or "",
                    is_verified=user.is_verified,
                    mfa_enabled=user.mfa_enabled,
                    created_at=created_at,
                    last_login=last_login,
                    roles=[user.tier],
                )

                await centralized_cache_service.set_user_cached(token_data.user_id, MessageToDict(user_info))
                return user_info
        except Exception as e:
            logger.error("grpc_get_user_info_failed", error=str(e))
            handle_grpc_error(e, context)
            return auth_pb2.UserInfo()

    async def CreateTokenPair(self, request, context):
        try:
            token_pair = auth_service.create_token_pair(
                user_id=request.user_id,
                email=request.email,
                tier=request.tier,
                scopes=list(request.scopes) if hasattr(request, "scopes") else [],
            )

            issued_at = timestamp_pb2.Timestamp()
            issued_at.GetCurrentTime()

            return auth_pb2.TokenPairResponse(
                access_token=token_pair.access_token,
                refresh_token=token_pair.refresh_token,
                token_type=token_pair.token_type,
                expires_in=token_pair.expires_in,
                issued_at=issued_at,
            )
        except Exception as e:
            logger.error("grpc_create_token_pair_failed", error=str(e))
            handle_grpc_error(e, context)
            return auth_pb2.TokenPairResponse()

    async def ValidateAPIKey(self, request, context):
        try:
            key_hash = hashlib.sha256(request.api_key.encode()).hexdigest()

            cached_api_key_data = await centralized_cache_service.get_api_key_cached(key_hash)
            if cached_api_key_data:
                await centralized_cache_service.update_api_key_last_used(key_hash)
                api_key_resp = auth_pb2.APIKeyResponse()
                ParseDict(cached_api_key_data, api_key_resp)
                return api_key_resp

            async with db_manager.async_session_factory() as db:
                result = await db.execute(
                    select(APIKey)
                    .options(joinedload(APIKey.user))
                    .where(APIKey.key_hash == key_hash, APIKey.is_active)
                )
                key_record = result.scalar_one_or_none()
                if not key_record:
                    return auth_pb2.APIKeyResponse(valid=False)

                user = key_record.user
                created_at = timestamp_pb2.Timestamp()
                created_at.FromDatetime(key_record.created_at)

                api_key_resp = auth_pb2.APIKeyResponse(
                    valid=True,
                    user_id=str(user.id),
                    email=user.email,
                    tier=user.tier,
                    key_name=key_record.name or "",
                    created_at=created_at,
                )

                await centralized_cache_service.set_api_key_cached(key_hash, MessageToDict(api_key_resp))
                await centralized_cache_service.update_api_key_last_used(key_hash)
                return api_key_resp
        except Exception as e:
            logger.error("grpc_validate_api_key_failed", error=str(e))
            handle_grpc_error(e, context)
            return auth_pb2.APIKeyResponse(valid=False)

    async def IntrospectToken(self, request, context):
        try:
            token_data = await auth_service.validate_token(request.token)
            return auth_pb2.IntrospectionResponse(
                active=True,
                sub=token_data.user_id,
                username=token_data.email,
                token_type=token_data.token_type,
                exp=int(token_data.exp.timestamp()),
                iat=int(token_data.iat.timestamp()),
                scope=" ".join(token_data.scopes),
                iss="manifold-auth-v2",
            )
        except Exception:
            return auth_pb2.IntrospectionResponse(active=False)


async def serve(port: str = "50051"):
    options = [
        ("grpc.keepalive_time_ms", 30000),
        ("grpc.keepalive_timeout_ms", 10000),
        ("grpc.keepalive_permit_without_calls", True),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),
    ]

    server = grpc.aio.server(options=options)
    auth_servicer = AuthServicer()
    auth_pb2_grpc.add_AuthServiceServicer_to_server(auth_servicer, server)

    health_servicer = health.HealthServicer()
    health_servicer.set("auth.AuthService", health_pb2.HealthCheckResponse.SERVING)
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    listen_addr = f"0.0.0.0:{port}"

    SECURE_MODE = os.getenv("GRPC_SECURE", "true").lower() == "true"
    
    try:
        # Improved cert path handling: prioritize ENV then standard paths
        ca_cert = os.getenv("GRPC_CA_CERT", "/etc/pki/root_ca.crt")
        server_crt = os.getenv("GRPC_SERVER_CERT", "/etc/pki/auth-service.crt")
        server_key = os.getenv("GRPC_SERVER_KEY", "/etc/pki/auth-service.key")

        if SECURE_MODE and all(os.path.exists(p) for p in [ca_cert, server_crt, server_key]):
            with open(server_key, "rb") as f:
                private_key = f.read()
            with open(server_crt, "rb") as f:
                certificate_chain = f.read()
            with open(ca_cert, "rb") as f:
                root_certificates = f.read()

            server_credentials = grpc.ssl_server_credentials(
                [(private_key, certificate_chain)],
                root_certificates=root_certificates,
                require_client_auth=True,
            )
            server.add_secure_port(listen_addr, server_credentials)
            logger.info("grpc_auth_server_starting_secure", port=port, mtls=True)
        else:
            server.add_insecure_port(listen_addr)
            logger.warning("grpc_auth_server_starting_insecure", port=port, reason="Certs missing or SECURE_MODE=false")
    except Exception as e:
        logger.exception("grpc_setup_failed", error=str(e))
        server.add_insecure_port(listen_addr)

    await server.start()
    logger.info("grpc_server_online", addr=listen_addr)
    await server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())