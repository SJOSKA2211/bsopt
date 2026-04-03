import asyncio
import logging
import os
from datetime import UTC, datetime

import grpc
import structlog
from cachetools import TTLCache
from google.protobuf import empty_pb2, timestamp_pb2
from google.protobuf.json_format import MessageToDict, ParseDict
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from sqlalchemy import select

from src.auth.auth import auth_service
from src.database import db_manager
from src.database.models import APIKey, User
from src.protos import auth_pb2, auth_pb2_grpc
from src.shared.utils.cache import db_cache

logger = structlog.get_logger(__name__)


class AuthServicer(auth_pb2_grpc.AuthServiceServicer):
    """
    Consolidated High-Performance Auth gRPC Servicer.
    Implements all methods defined in auth.proto for zero-trust internal comms.
    Distributed caching via Redis (db_cache) ensures consistency across Manifold instances.
    """

    def __init__(self):
        self._user_cache = TTLCache(maxsize=10000, ttl=300)
        self._api_key_cache = TTLCache(maxsize=10000, ttl=300)

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

            # We return the new access token info
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
            return auth_pb2.TokenResponse(valid=False)

    async def RevokeToken(self, request, context):
        try:
            await auth_service.revoke_token(request.token)
            return empty_pb2.Empty()
        except Exception as e:
            logger.error("grpc_revoke_token_failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return empty_pb2.Empty()

    async def GetUserInfo(self, request, context):
        try:
            token_data = await auth_service.validate_token(request.token)

            # Local In-Memory Cache
            if token_data.user_id in self._user_cache:
                logger.debug("user_info_local_cache_hit", user_id=token_data.user_id)
                return self._user_cache[token_data.user_id]

            # Check Distributed Cache
            cached_data = await db_cache.get_user(token_data.user_id)
            if cached_data:
                logger.debug("user_info_cache_hit", user_id=token_data.user_id)
                user_info = auth_pb2.UserInfo()
                ParseDict(cached_data, user_info)
                self._user_cache[token_data.user_id] = user_info
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

                # Update Cache (Convert to Dict for Redis storage)
                await db_cache.set_user(token_data.user_id, MessageToDict(user_info))
                self._user_cache[token_data.user_id] = user_info
                return user_info
        except Exception as e:
            logger.error("grpc_get_user_info_failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
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
            context.set_code(grpc.StatusCode.INTERNAL)
            return auth_pb2.TokenPairResponse()

    async def ValidateAPIKey(self, request, context):
        try:
            import hashlib

            key_hash = hashlib.sha256(request.api_key.encode()).hexdigest()

            # Local In-Memory Cache
            if key_hash in self._api_key_cache:
                logger.debug("api_key_local_cache_hit", key_hash=key_hash[:10] + "...")
                # We can skip updating the last_used_at in Redis synchronously for max perf,
                # or buffer it, but we'll stick to serving from cache.
                return self._api_key_cache[key_hash]

            # Check Distributed Cache
            cached_data = await db_cache.get_api_key(key_hash)
            if cached_data:
                logger.debug("api_key_cache_hit", key_hash=key_hash[:10] + "...")
                # Buffer the last_used_at update in Redis
                from src.shared.utils.cache import get_redis

                redis = get_redis()
                if redis:
                    await redis.hset("api_key_last_used", key_hash, datetime.now(UTC).isoformat())

                api_key_resp = auth_pb2.APIKeyResponse()
                ParseDict(cached_data, api_key_resp)
                self._api_key_cache[key_hash] = api_key_resp
                return api_key_resp

            async with db_manager.async_session_factory() as db:
                # Optimized join query to avoid lazy loading of user
                from sqlalchemy.orm import joinedload

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

                # Update Cache
                await db_cache.set_api_key(key_hash, MessageToDict(api_key_resp))
                self._api_key_cache[key_hash] = api_key_resp
                return api_key_resp
        except Exception as e:
            logger.error("grpc_validate_api_key_failed", error=str(e))
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
    server = grpc.aio.server()
    auth_servicer = AuthServicer()
    auth_pb2_grpc.add_AuthServiceServicer_to_server(auth_servicer, server)

    # Add standard Health Checking Service
    health_servicer = health.HealthServicer()
    health_servicer.set("auth.AuthService", health_pb2.HealthCheckResponse.SERVING)
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    listen_addr = f"0.0.0.0:{port}"

    # Secure gRPC implementation
    try:
        PROJECT_ROOT = os.getcwd()
        with open(os.path.join(PROJECT_ROOT, ".pki/auth-service.key"), "rb") as f:
            private_key = f.read()
        with open(os.path.join(PROJECT_ROOT, ".pki/auth-service.crt"), "rb") as f:
            certificate_chain = f.read()
        with open(os.path.join(PROJECT_ROOT, ".pki/root_ca.crt"), "rb") as f:
            root_certificates = f.read()

        server_credentials = grpc.ssl_server_credentials(
            [(private_key, certificate_chain)],
            root_certificates=root_certificates,
            require_client_auth=True,
        )
        server.add_secure_port(listen_addr, server_credentials)
        logger.info("grpc_auth_server_starting_secure", port=port)
    except Exception as e:
        logger.error("grpc_secure_setup_failed_falling_back", error=str(e))
        server.add_insecure_port(listen_addr)
        logger.info("grpc_auth_server_starting_insecure", port=port)

    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())
