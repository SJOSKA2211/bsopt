import asyncio
import logging
from datetime import UTC, datetime

import grpc
from google.protobuf import timestamp_pb2, empty_pb2

from src.auth.auth import auth_service
from src.protos import auth_pb2, auth_pb2_grpc
from src.database import db_manager
from src.database.models import User, APIKey
from sqlalchemy import select

logger = logging.getLogger(__name__)

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
                roles=token_data.scopes
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
                scopes=token_data.scopes
            )
            
            # We return the new access token info
            # Decoding it to get the exact exp/iat
            new_access_data = auth_service.decode_token(token_pair.access_token)

            return auth_pb2.TokenResponse(
                valid=True,
                user_id=new_access_data.user_id,
                email=new_access_data.email,
                tier=new_access_data.tier,
                expires_at=int(new_access_data.exp.timestamp()),
                issued_at=int(new_access_data.iat.timestamp()),
                token_type="access",
                roles=new_access_data.scopes
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

                return auth_pb2.UserInfo(
                    user_id=str(user.id),
                    email=user.email,
                    tier=user.tier,
                    full_name=user.full_name or "",
                    is_verified=user.is_verified,
                    mfa_enabled=user.mfa_enabled,
                    created_at=created_at,
                    last_login=last_login,
                    roles=[user.tier]
                )
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
                scopes=list(request.scopes) if hasattr(request, 'scopes') else []
            )
            
            issued_at = timestamp_pb2.Timestamp()
            issued_at.GetCurrentTime()

            return auth_pb2.TokenPairResponse(
                access_token=token_pair.access_token,
                refresh_token=token_pair.refresh_token,
                token_type=token_pair.token_type,
                expires_in=token_pair.expires_in,
                issued_at=issued_at
            )
        except Exception as e:
            logger.error("grpc_create_token_pair_failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return auth_pb2.TokenPairResponse()

    async def ValidateAPIKey(self, request, context):
        try:
            import hashlib
            key_hash = hashlib.sha256(request.api_key.encode()).hexdigest()
            async with db_manager.async_session_factory() as db:
                result = await db.execute(
                    select(APIKey).where(APIKey.key_hash == key_hash, APIKey.is_active)
                )
                key_record = result.scalar_one_or_none()
                if not key_record:
                    return auth_pb2.APIKeyResponse(valid=False)
                
                user = key_record.user
                created_at = timestamp_pb2.Timestamp()
                created_at.FromDatetime(key_record.created_at)
                
                return auth_pb2.APIKeyResponse(
                    valid=True,
                    user_id=str(user.id),
                    email=user.email,
                    tier=user.tier,
                    key_name=key_record.name or "",
                    created_at=created_at
                )
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
                iss="manifold-auth-v2"
            )
        except Exception:
            return auth_pb2.IntrospectionResponse(active=False)

async def serve(port: str = "50051"):
    server = grpc.aio.server()
    auth_pb2_grpc.add_AuthServiceServicer_to_server(AuthServicer(), server)
    listen_addr = f"0.0.0.0:{port}"
    server.add_insecure_port(listen_addr)
    logger.info("grpc_auth_server_starting", port=port)
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())
