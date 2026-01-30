# BS-Opt Security and Hardening Protocol

This document outlines the security measures and platform hardening steps implemented to protect the BS-Opt platform.

## 1. Zero Trust Architecture
- **mTLS**: Mandatory Mutual TLS for all service-to-service communication.
- **OPA**: Open Policy Agent enforced for fine-grained access control.
- **API Key Security**: API keys are hashed using SHA-256 and stored securely.

## 2. Authentication and MFA
- **Encryption at Rest**: MFA secrets are encrypted with `Fernet` (AES-128 in CBC mode with HMAC SHA256).
- **Hashed Backup Codes**: MFA backup codes are stored as SHA-256 hashes.
- **Timing Attack Protection**: Constant-time comparison implemented in `AuthService.authenticate_user`.

## 3. Webhook Security
- **HMAC Signatures**: All incoming webhooks must include a `X-Webhook-Signature` header.
- **Verification**: Signatures are verified using a shared secret and HMAC-SHA256 over the raw request body.

## 4. Platform Hardening
- **Infrastructure**: Docker containers use `cpuset` for core pinning and resource isolation.
- **Logging**: PII (like Client IPs) is automatically masked in logs.
- **Concurrency**: Local builds are limited to 1 worker process to prevent system freezing (see `Anti-Freeze Guide`).

## 5. Continuous Security
- **Scheduled Scans**: `app-pipeline.yml` runs `pip-audit` and `bandit` on every push.
- **Daily Training**: `mlops-training.yml` performs daily model retraining and performance verification.

---
*Maintained by the Security and DevOps Teams.*
