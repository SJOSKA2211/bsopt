# Design Document: Phase 0 - Zero-Touch Infrastructure & Auth Bootstrapping

**Date**: 2026-03-20
**Topic**: Manifold Phase 0 Infrastructure & Auth
**Status**: DRAFT (v2.0 - Revised after Review)
**Task Complexity**: Complex

## 1. Executive Summary
This design defines Phase 0 of the Manifold platform, focusing on automated, zero-touch infrastructure provisioning and a zero-trust authentication boundary. The goal is to move from a standard Docker setup to a cryptographically secured, Production-grade stack that is container-engine agnostic and self-healing.

## 2. Problem Statement
Current infrastructure lacks:
- Automated asymmetric key management for both identity and secret storage.
- Strict mTLS boundaries between internal microservices.
- A unified, container-agnostic bootstrap process that handles health gating and performance tuning.
- A production-hardened API gateway with native rate-limiting and circuit-breaking.

## 3. Proposed Architecture (Cryptographic Vault)

### 3.1. Security & Identity Layer
- **Root CA**: An internal Certificate Authority (CA) will be generated during bootstrap.
- **Asymmetric JWTs**: 
    - **ECC P-256 (ES256)**: Primary for high-performance session validation (WebSockets).
    - **RSA 4096 (RS256)**: Secondary for cross-platform compatibility.
- **Argon2id**: Standardized hashing for all passwords with legacy migration logic.
- **Vaulting & Decryption Shim**: 
    - Sensitive `.env` values are encrypted using an RSA 4096 "Vault Key".
    - **Implementation**: A lightweight shell-based decryption shim (`entrypoint.sh`) will use `openssl pkeyutl` to decrypt secrets into environment variables at runtime, ensuring plaintext secrets never hit the disk or `docker inspect` output.

### 3.2. Container-Agnostic Bootstrapping
- **`bootstrap.sh` (v3.1)**:
    - Detects `podman` vs `docker` and aliases `COMPOSE` commands.
    - Generates `.pki/` hierarchy (Root CA, Server/Client Certs, JWT Keys).
    - **Health Gating Sequence**:
        1. **Postgres**: `pg_isready -U admin`
        2. **Redis**: `redis-cli ping`
        3. **RabbitMQ**: `rabbitmq-diagnostics -q check_running`
        4. **Auth Service**: `wget -qO- http://localhost:3001/health`
        5. **API Service**: `wget -qO- http://localhost:8000/health`
    - **DB Tuning**: Injects NVMe-optimized `postgresql.conf` and `sysctl` parameters.

### 3.3. Database Infrastructure (TimescaleDB + PgBouncer)
- **Primary**: PostgreSQL 16 + TimescaleDB extension.
- **mTLS Enforcement**:
    - `pg_hba.conf` will be updated to:
      `hostssl bsopt all 0.0.0.0/0 cert clientcert=verify-full`
    - This forces all internal service connections to present a valid certificate signed by the internal Root CA.
- **Pooling**: PgBouncer in `transaction` mode. `max_client_conn` set to 2000; `default_pool_size` at 50.

### 3.4. Edge Gateway (Envoy)
- **mTLS Termination**: Envoy terminates external TLS 1.3 and uses internal mTLS for upstream communication.
- **Rate Limiting**: Local token-bucket rate limiting (100 req/s per IP).
- **Circuit Breaking**: Outlier detection to eject failing service instances.

## 4. Components & Data Flow

1. **Bootstrap Phase**: Shell script -> OpenSSL -> File System (`.pki`, `.env`).
2. **Launch Phase**: `docker-compose` -> Image Builds -> Container Start.
3. **Entrypoint Phase**: `entrypoint.sh` -> RSA Decrypt -> Set ENV -> Exec Main Process.
4. **Auth Flow**: User -> Envoy -> Auth Service (Argon2id) -> Signed ES256 JWT -> Envoy (JWT Validation Filter) -> Downstream Services.

## 5. Technical Decisions & Tradeoffs

- **Decision**: Use `ECC P-256` for JWTs.
    - **Tradeoff**: Smaller signatures and faster verification than RSA.
- **Decision**: mTLS for Internal Traffic.
    - **Tradeoff**: Increases configuration complexity for all clients, but provides cryptographic proof of service identity.

## 6. Validation & Testing
- **mTLS Verification**: `openssl s_client` connection to Postgres port from within an app container.
- **Vault Verification**: `env` check within container to ensure decrypted secrets are present but absent in `docker inspect`.
- **Bootstrap Gauntlet**: All services must reach "Healthy" status in under 120s.

## 7. Next Steps
1. Update `infrastructure/configs/pg_hba.conf` for mTLS.
2. Modify `bootstrap.sh` with the expanded health gating loop.
3. Create `infrastructure/scripts/entrypoint-shim.sh` for RSA decryption.
4. Update Dockerfiles to incorporate the new entrypoint shim.
