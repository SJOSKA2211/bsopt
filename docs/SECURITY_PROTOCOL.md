# BS-Opt Security and Hardening Protocol

This document outlines the security measures and platform hardening steps implemented to protect the BS-Opt platform.

## 1. Zero Trust Architecture
- **mTLS**: Mandatory Mutual TLS for all service-to-service communication.
- **OPA**: Open Policy Agent enforced for fine-grained access control.
- **API Key Security**: API keys are hashed using SHA-256 and stored securely.

## 2. Authentication and MFA
- **Centralized Auth**: All authentication flows (login, register, MFA) are centralized in the Node.js `auth-service` using `better-auth`.
- **Argon2id Hashing**: Standardized on Argon2id for all password hashing (Memory: 64MB, Time: 3, Parallelism: 4).
- **Session Verification**: Python backend verifies `better_auth_sessions` with Redis caching (5-minute TTL) for performance.
- **MFA (Two-Factor)**: MFA is enforced via the `better-auth` two-factor plugin, replacing the custom legacy implementation.
- **Rate Limiting**: Distributed sliding window rate limiting implemented in Redis LUA for both Node.js and Python src.
- **Timing Attack Protection**: Constant-time comparison and dummy hashing implemented in legacy `AuthService.authenticate_user`.

## 3. Webhook Security
- **HMAC Signatures**: All incoming webhooks must include a `X-Webhook-Signature` header.
- **Verification**: Signatures are verified using a shared secret and HMAC-SHA256 over the raw request body.

## 4. Platform Hardening
- **Infrastructure**: Docker containers use `cpuset` for src.shared pinning and resource isolation.
- **Logging**: PII (like Client IPs) is automatically masked in logs.
- **Concurrency**: Local builds are limited to 1 worker process to prevent system freezing (see `Anti-Freeze Guide`).

## 5. Continuous Security
- **Scheduled Scans**: `app-pipeline.yml` runs `pip-audit` and `bandit` on every push.
- **Daily Training**: `mlops-training.yml` performs daily model retraining and performance verification.

---
*Maintained by the Security and DevOps Teams.*
