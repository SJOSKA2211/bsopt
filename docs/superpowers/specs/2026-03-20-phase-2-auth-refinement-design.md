# Design Document: Phase 2 - Single-Source Zero-Trust Auth & Backend Refinement

**Date**: 2026-03-20
**Topic**: Unified Auth & Zero-Trust Middleware
**Status**: DRAFT (v2.0 - Revised after Review)
**Task Complexity**: Complex

## 1. Executive Summary
Phase 2 focuses on consolidating the EquaFlow security perimeter. By unifying authentication logic into a single high-performance service and enforcing a "fused" zero-trust middleware (JWT + mTLS) across all microservices, we ensure that every internal request is cryptographically verified for both user and service identity.

## 2. Problem Statement
- Auth logic is currently fragmented across multiple modules (`password.py`, `oauth2.py`, `mfa.py`).
- Internal service-to-service communication relies on mTLS at the network level, but often lacks application-level identity verification (JWT).
- Some schemas are still using Pydantic V1 patterns, missing the performance gains of V2's Rust core.

## 3. Proposed Architecture (Unified Security Mesh)

### 3.1. Consolidated Zero-Trust Auth Service
- **Unified `AuthService`**: A single service in `src/auth/auth.py` that orchestrates:
    - **Argon2id**: High-entropy password hashing and verification.
    - **MFA**: TOTP-based multi-factor authentication.
    - **Token Management**: Issuance and revocation of ES256 (ECC) and RS256 (RSA) JWTs.
- **gRPC Interface**: Implements the `equaflow.auth.AuthService` protocol defined in `protos/auth.proto` for low-latency inter-service validation.

### 3.2. Fused Zero-Trust Middleware
- **Security Gating**: A mandatory `ZeroTrustMiddleware` in `src/api/middleware/fused.py`.
- **Dual-Factor Verification**:
    1. **Identity (mTLS)**: Verifies `X-SSL-Client-Verify` and `X-SSL-Client-S-DN` headers.
    2. **Permission (JWT)**: Validates asymmetric JWT signatures.
- **Proxy Trust**: The middleware strictly enforces that `X-SSL-*` headers are ONLY accepted from IPs in the `TRUSTED_PROXIES` list (e.g., Envoy's internal IP). External requests with spoofed headers are rejected.
- **Fail-Fast**: Internal routes (e.g., `/api/internal/*`) explicitly fail if mTLS headers are missing, even if a valid JWT is present.

### 3.3. Performance Schema Standard (Hybrid Approach)
- **Pattern**: Standardize on a hybrid **msgspec + Pydantic V2** pattern for all high-throughput schemas.
    - **msgspec**: Used for zero-copy, ultra-fast JSON serialization/deserialization.
    - **Pydantic V2**: Used for complex validation, coercion, and business logic rules via `model_validator`.
- **Protobuf-to-Pydantic Bridge**: Implement `from_proto` and `to_proto` class methods on core Pydantic models to automate mapping between gRPC messages and application-layer models.

## 4. Components & Data Flow
1. **Login**: User -> Envoy -> Auth Service (FastAPI) -> (Argon2id/MFA) -> Signed ES256 JWT.
2. **Service Call**: API Service -> gRPC (ValidateToken) -> Auth Service (gRPC).
3. **Internal Request**: Service A -> Service B (Middleware validates mTLS Header + JWT).

## 5. Technical Decisions & Tradeoffs
- **Decision**: Mandatory JWT + mTLS for internal traffic.
    - **Tradeoff**: Increases latency by ~1-2ms per hop, but provides defense-in-depth against lateral movement.
- **Decision**: Standardization on ES256 for internal tokens.
    - **Tradeoff**: Faster verification and smaller headers than RSA.

## 6. Validation & Testing
- **Security Gauntlet**: Attempt service calls with valid mTLS but invalid JWT (should fail).
- **Spoof Test**: Attempt a request with `X-SSL-Client-Verify` from an untrusted IP (should fail).
- **Benchmark**: Verify that the msgspec+Pydantic hybrid maintains < 5ms total overhead per request under load.

## 7. Next Steps
1. Refactor `src/auth/auth.py` to consolidate all auth logic.
2. Implement the `ZeroTrustMiddleware` with proxy IP verification in `fused.py`.
3. Systematically apply the hybrid schema pattern to `src/api/schemas/`.
