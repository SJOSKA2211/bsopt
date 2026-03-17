# BS-OPT Containerized Development Environment Guide

## Prerequisites
- Docker and Docker Compose installed.
- `.env` file configured with necessary secrets (see `.env.example`).

## Quick Start
To start the entire development environment:
```bash
docker compose -f docker-compose.yml up --build
```

## Services
| Service | Internal Port | External Port | Description |
| :--- | :--- | :--- | :--- |
| `api` | 8000 | 8000 | Core Python API (FastAPI) |
| `auth-service` | 3001 | 3001 | Better Auth Microservice (Node.js) |
| `frontend` | 5173 | 5173 | React Application (Vite) |
| `neural-pricing` | 8000 | 8001 | Neural Pricing Engine (FastAPI) |
| `postgres` | 5432 | 5432 | Database |
| `redis` | 6379 | 6379 | Cache/Task Broker |
| `rabbitmq` | 5672 | 5672 | Task Broker |

## CLI Tool (Containerized)
You can run the CLI tool without a local Python environment using the Makefile:
```bash
# Build the CLI image first
make build-cli

# Run a CLI command
make cli ARGS="--help"

# Example: Check system status
make cli ARGS="status"
```

## Hot-Reloading
- **Frontend**: Enabled by default via Vite. Polling is enabled for compatibility with Docker volumes.
- **API/Neural Pricing**: Enabled via Uvicorn `--reload`.
- **Auth Service**: Enabled via `tsx watch`.

## Debugging
### Frontend
- Source maps are enabled by default in development mode.
- Use Chrome DevTools or VS Code Debugger (attach to `http://localhost:5173`).

### Backend (Python)
- To enable debugging, you can use `debugpy`. (Not configured by default, but can be added to the dev Dockerfiles).

## High-Performance Mode (Silicon Lockdown)
To enable ultra-low latency features (AF_XDP Ingestion and CPU Pinning), the containers require additional Linux capabilities.

Add the following to your `docker-compose.override.yml`:
```yaml
services:
  worker:
    cap_add:
      - NET_RAW
      - SYS_NICE
    # For core pinning to work correctly, use host networking if possible
    # network_mode: host 
```

### Silicon Lockdown Core Mapping:
- **Core 1**: Dedicated to `XDPIngester` (High-priority thread).
- **Core 2**: Dedicated to `OnlineRLAgent` (Spinning inference loop).

## Troubleshooting
- **Permission Denied**: If you get `permission denied` on `docker.sock`, ensure your user is in the `docker` group or use `sudo`.
- **Database Connection**: Ensure the `DATABASE_URL` in `.env` points to `postgres` service when running inside Docker.
