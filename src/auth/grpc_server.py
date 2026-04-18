import asyncio
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any, Optional

import grpc
from google.protobuf import empty_pb2
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from src.auth import auth
from src.database.crud import get_user_by_id
from src.database.session import engine as db_engine
from src.shared.protos import auth_pb2, auth_pb2_grpc

# --- Configuration ---
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

JWT_SECRET_KEY = os.getenv("JWT_SECRET", "super-dev-secret-change-me-in-prod")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))

# --- TLS Configuration ---
SERVER_CERT_PATH = "/etc/ssl/certs/auth_service.crt"
SERVER_KEY_PATH = "/etc/ssl/private/auth_service.key"
CA_CERT_PATH = "/etc/ssl/certs/root_ca.crt"

class AuthServicer(auth_pb2_grpc.AuthServiceServicer):
    """gRPC servicer for authentication and authorization."""

    async def ValidateToken(
        self, 
        request: auth_pb2.ValidateTokenRequest, 
        context: grpc.aio.ServicerContext
    ) -> auth_pb2.TokenResponse:
        """Validates a JWT token and returns user information."""
        token = request.token
        logger.info("Validating token (prefix: %s)", token[:10])

        payload = auth.verify_token(token)
        if payload is None:
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Token is invalid or expired")

        async with AsyncSession(db_engine) as db:
            user_id_str = payload.get("sub")
            if not user_id_str:
                await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Missing subject in token")
            
            user = await get_user_by_id(db, user_id=UUID(user_id_str))
            if not user:
                await context.abort(grpc.StatusCode.UNAUTHENTICATED, "User not found")
                return auth_pb2.TokenResponse()

        return auth_pb2.TokenResponse(
            valid=True,
            user_id=str(user.id),
            email=user.email,
            tier=user.tier,
            expires_at=int(payload.get("exp", 0)),
            issued_at=int(payload.get("iat", 0)),
            token_type=payload.get("token_type", "access"),
            roles=user.roles if isinstance(user.roles, list) else [],
        )

    async def CreateTokenPair(
        self, 
        request: auth_pb2.TokenPairRequest, 
        context: grpc.aio.ServicerContext
    ) -> auth_pb2.TokenPairResponse:
        """Creates an access/refresh token pair for a user."""
        logger.info("Creating token pair for user_id: %s", request.user_id)

        access_payload = {
            "sub": request.user_id,
            "email": request.email,
            "tier": request.tier,
            "roles": list(request.scopes),
            "token_type": "access",
        }
        refresh_payload = {
            "sub": request.user_id,
            "email": request.email,
            "token_type": "refresh",
        }

        access_token = auth.create_access_token(access_payload)
        refresh_token = auth.create_refresh_token(refresh_payload)

        return auth_pb2.TokenPairResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            issued_at=datetime.now(UTC).isoformat(),
        )

    async def RefreshToken(
        self, 
        request: auth_pb2.RefreshTokenRequest, 
        context: grpc.aio.ServicerContext
    ) -> auth_pb2.TokenPairResponse:
        """Refreshes an access token using a valid refresh token."""
        logger.info("Refreshing token (prefix: %s)", request.refresh_token[:10])

        payload = auth.verify_token(request.refresh_token)
        if payload is None or payload.get("token_type") != "refresh":
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid or expired refresh token")

        async with AsyncSession(db_engine) as db:
            user_id_str = payload.get("sub")
            if not user_id_str:
                await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid token payload")

            user = await get_user_by_id(db, user_id=UUID(user_id_str))
            if not user:
                await context.abort(grpc.StatusCode.UNAUTHENTICATED, "User not found")

        access_payload = {
            "sub": str(user.id),
            "email": user.email,
            "tier": user.tier,
            "roles": user.roles if isinstance(user.roles, list) else [],
            "token_type": "access",
        }
        new_access_token = auth.create_access_token(access_payload)
        new_refresh_token = auth.create_refresh_token({"sub": str(user.id), "token_type": "refresh"})

        return auth_pb2.TokenPairResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            issued_at=datetime.now(UTC).isoformat(),
        )

    async def RevokeToken(
        self, 
        request: auth_pb2.RevokeTokenRequest, 
        context: grpc.aio.ServicerContext
    ) -> empty_pb2.Empty:
        """Revokes a token."""
        logger.info("Revoking token (prefix: %s)", request.token[:10])
        auth.revoke_token(request.token)
        return empty_pb2.Empty()

    async def GetUserInfo(
        self, 
        request: auth_pb2.GetUserInfoRequest, 
        context: grpc.aio.ServicerContext
    ) -> auth_pb2.UserInfo:
        """Retrieves detailed user information from a token."""
        logger.info("Getting user info for token prefix: %s", request.token[:10])
        payload = auth.verify_token(request.token)
        if payload is None:
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Token is invalid or expired")

        async with AsyncSession(db_engine) as db:
            user_id_str = payload.get("sub")
            if not user_id_str:
                await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid token payload")

            user = await get_user_by_id(db, user_id=UUID(user_id_str))
            if not user:
                await context.abort(grpc.StatusCode.UNAUTHENTICATED, "User not found")

            return auth_pb2.UserInfo(
                user_id=str(user.id),
                email=user.email,
                full_name=user.full_name or "",
                tier=user.tier,
                is_verified=user.is_verified,
                mfa_enabled=user.mfa_enabled,
                created_at=user.created_at.timestamp() if user.created_at else 0,
                last_login=user.last_login.timestamp() if user.last_login else 0,
                roles=user.roles if isinstance(user.roles, list) else [],
            )

    async def ValidateAPIKey(self, request: Any, context: grpc.aio.ServicerContext) -> Any:
        """Validates an API key (Unimplemented)."""
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, "Method not implemented")

    async def IntrospectToken(
        self, 
        request: auth_pb2.IntrospectTokenRequest, 
        context: grpc.aio.ServicerContext
    ) -> auth_pb2.IntrospectionResponse:
        """Introspects a token to check its status."""
        logger.info("Introspecting token prefix: %s", request.token[:10])
        payload = auth.verify_token(request.token)

        if payload is None:
            return auth_pb2.IntrospectionResponse(active=False)

        return auth_pb2.IntrospectionResponse(
            active=True,
            sub=payload.get("sub"),
            username=payload.get("email"),
            token_type=payload.get("token_type"),
            exp=int(payload.get("exp", 0)),
            iat=int(payload.get("iat", 0)),
            scope=" ".join(payload.get("roles", [])),
            iss="ManifoldAuth",
        )

def load_tls_credentials() -> Optional[grpc.ServerCredentials]:
    """Loads TLS credentials from disk."""
    try:
        with open(CA_CERT_PATH, "rb") as f: root_certs = f.read()
        with open(SERVER_CERT_PATH, "rb") as f: server_cert = f.read()
        with open(SERVER_KEY_PATH, "rb") as f: server_key = f.read()
        return grpc.ssl_server_credentials(((server_cert, server_key),), root_certs, True)
    except FileNotFoundError as e:
        logger.error("TLS certificate file not found: %s", e)
    except Exception as e:
        logger.exception("Error loading TLS certificates: %s", e)
    return None

async def serve() -> None:
    """Starts the gRPC server."""
    server = grpc.aio.server()
    auth_pb2_grpc.add_AuthServiceServicer_to_server(AuthServicer(), server)

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)

    listen_addr = "[::]:50051"
    
    server_credentials = load_tls_credentials()
    if server_credentials:
        server.add_secure_port(listen_addr, server_credentials)
        logger.info("Starting gRPC server on %s with TLS enabled.", listen_addr)
    else:
        server.add_insecure_port(listen_addr)
        logger.warning("Starting gRPC server on %s WITHOUT TLS (Insecure Mode).", listen_addr)

    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())

if __name__ == "__main__":
    os.environ.setdefault("JWT_SECRET", "super-dev-secret-change-me-in-prod")
    os.environ.setdefault("ACCESS_TOKEN_EXPIRE_MINUTES", "15")
    os.environ.setdefault("REFRESH_TOKEN_EXPIRE_DAYS", "30")
    os.environ.setdefault("MFA_ENCRYPTION_KEY", "a-very-strong-mfa-secret-key-for-dev")

    asyncio.run(serve())
