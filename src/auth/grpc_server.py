import asyncio
import logging
import os
from datetime import UTC, datetime, timedelta

import grpc
from google.protobuf import empty_pb2
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from sqlalchemy.ext.asyncio import AsyncSession  # For type hinting

from src.auth import auth  # Import the auth module
from src.database.crud import (  # Assuming these CRUD functions exist or will be added
    get_user_by_id,
)
from src.database.session import engine as db_engine  # Import the global async engine
from src.shared.protos import auth_pb2, auth_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
JWT_SECRET_KEY = os.getenv("JWT_SECRET", "my-super-secret-key-for-development-only")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "RS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))
MFA_ENCRYPTION_KEY = os.getenv("MFA_ENCRYPTION_KEY", "my-mfa-secret-key-for-development-only")

# --- Token Revocation Store (In-memory, not production-ready) ---
REVOKED_TOKENS = set()

# --- TLS Configuration ---
SERVER_CERT_PATH = "/etc/ssl/certs/auth_service.crt"
SERVER_KEY_PATH = "/etc/ssl/private/auth_service.key"
CA_CERT_PATH = "/etc/ssl/certs/root_ca.crt"

class AuthServicer(auth_pb2_grpc.AuthServiceServicer):
    async def ValidateToken(self, request, context):
        token = request.token
        logger.info(f"Validating token: {token[:10]}...")

        if token in REVOKED_TOKENS:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Token has been revoked")

        payload = auth.verify_token(token)

        if payload is None:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Token is invalid or expired")

        async with AsyncSession(db_engine) as db:
            user = await get_user_by_id(db, user_id=payload.get("sub"))
            if not user:
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "User not found for this token")
                return auth_pb2.TokenResponse()

        return auth_pb2.TokenResponse(
            valid=True,
            user_id=user.id,
            email=user.email,
            tier=user.tier,
            expires_at=int(payload.get("exp", datetime.now(UTC) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)).timestamp()),
            issued_at=int(payload.get("iat", datetime.now(UTC)).timestamp()),
            token_type=payload.get("token_type", "access"),
            roles=user.roles,
        )

    async def CreateTokenPair(self, request, context):
        logger.info(f"Creating token pair for user_id: {request.user_id}, email: {request.email}")

        access_payload = {
            "sub": request.user_id,
            "email": request.email,
            "tier": request.tier,
            "roles": request.scopes,
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
            issued_at=datetime.now(UTC),
        )

    async def RefreshToken(self, request, context):
        logger.info(f"Refreshing token: {request.refresh_token[:10]}...")

        payload = auth.verify_token(request.refresh_token)

        if payload is None or payload.get("token_type") != "refresh":
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid or expired refresh token")

        async with AsyncSession(db_engine) as db:
            user = await get_user_by_id(db, user_id=payload.get("sub"))
            if not user:
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "User not found for refresh token")

        access_payload = {
            "sub": user.id,
            "email": user.email,
            "tier": user.tier,
            "roles": user.roles,
            "token_type": "access",
        }
        new_access_token = auth.create_access_token(access_payload)

        new_refresh_token = auth.create_refresh_token({"sub": user.id, "token_type": "refresh"})

        return auth_pb2.TokenPairResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            issued_at=datetime.now(UTC),
        )

    async def RevokeToken(self, request, context):
        logger.info(f"Revoking token: {request.token[:10]}...")
        auth.revoke_token(request.token)
        # In a real system, also invalidate associated refresh tokens and persist revocation
        return empty_pb2.Empty()

    async def GetUserInfo(self, request, context):
        logger.info(f"Getting user info for token: {request.token[:10]}...")
        payload = auth.verify_token(request.token)
        if payload is None:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Token is invalid or expired")

        async with AsyncSession(db_engine) as db: # Use global engine for DB access
            user = await get_user_by_id(db, user_id=payload.get("sub"))
            if not user:
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "User not found for token")

            return auth_pb2.UserInfo(
                user_id=user.id,
                email=user.email,
                full_name=user.full_name,
                tier=user.tier,
                is_verified=user.is_verified,
                mfa_enabled=user.mfa_enabled,
                created_at=user.created_at.timestamp() if user.created_at else None,
                last_login=user.last_login.timestamp() if user.last_login else None,
                roles=user.roles,
            )

    async def ValidateAPIKey(self, request, context):
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Method not implemented")

    async def IntrospectToken(self, request, context):
        logger.info(f"Introspecting token: {request.token[:10]}...")
        token = request.token

        if token in REVOKED_TOKENS:
            return auth_pb2.IntrospectionResponse(active=False)

        payload = auth.verify_token(token)

        if payload is None:
            return auth_pb2.IntrospectionResponse(active=False)

        # Basic introspection: Check expiry and token type
        # More detailed introspection might involve checking issuer, audience, scopes, etc.
        # and potentially cross-referencing with DB or other stores.
        return auth_pb2.IntrospectionResponse(
            active=True,
            sub=payload.get("sub"),
            username=payload.get("email"),
            token_type=payload.get("token_type"),
            exp=int(payload.get("exp", 0)),
            iat=int(payload.get("iat", 0)),
            scope=payload.get("roles", []), # Assuming roles map to scope
            iss="ManifoldAuth", # Example issuer
        )

async def serve():
    server = grpc.aio.server()
    auth_pb2_grpc.add_AuthServiceServicer_to_server(AuthServicer(), server)

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)

    listen_addr = "[::]:50051"

    try:
        # Ensure the PKI directory is mounted and contains the necessary certs
        with open(CA_CERT_PATH, "rb") as f: root_certs = f.read()
        with open(SERVER_CERT_PATH, "rb") as f: server_cert = f.read()
        with open(SERVER_KEY_PATH, "rb") as f: server_key = f.read()
    except FileNotFoundError as e:
        logger.error(f"TLS certificate file not found: {e}. Ensure PKI files are mounted correctly at /etc/ssl/certs/ and /etc/ssl/private/. Server will not start.")
        return # Stop server startup
    except Exception as e:
        logger.error(f"Error loading TLS certificates: {e}. Server will not start.")
        return # Stop server startup

    server_credentials = grpc.ssl_server_credentials(
        ((server_cert, server_key),), root_certs, True,
    )

    server.add_secure_port(listen_addr, server_credentials)

    logger.info(f"Starting gRPC server on {listen_addr} with TLS enabled.")
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    os.environ.setdefault("JWT_SECRET", "super-dev-secret-change-me-in-prod")
    os.environ.setdefault("ACCESS_TOKEN_EXPIRE_MINUTES", "15")
    os.environ.setdefault("REFRESH_TOKEN_EXPIRE_DAYS", "30")
    os.environ.setdefault("MFA_ENCRYPTION_KEY", "a-very-strong-mfa-secret-key-for-dev")

    asyncio.run(serve())
