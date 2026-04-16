# Architecture: Hybrid Distributed Workers

## Overview
The BS-OPT platform uses a hybrid distributed architecture combining **Celery** for asynchronous task orchestration and **Ray** for high-performance compute-intensive tasks. This allows the system to handle thousands of background tasks (Webhooks, Audit, Scrapers) while delegating heavy math (Heston Calibration, Model Retraining) to a dedicated high-performance Ray cluster.

## Component Map
- **Task Broker**: Redis (v5.2.0) handles the message queuing for Celery.
- **Orchestration**: `Celery` (`src/workers/math_worker.py`) receives tasks from the API or other src.
- **Compute Engine**: `Ray` handles the actual computation via `MathActor` instances.

## Execution Flow: `recalibrate_symbol`
1. **Trigger**: An API request or a scheduled task calls `recalibrate_symbol` (`src/workers/math_worker.py:L58`).
2. **Celery Worker**: A Celery worker picks up the task.
3. **Ray Delegation**: The Celery worker identifies an available `MathActor` in the Ray ActorPool.
4. **Execution**: The `MathActor` performs the heavy calibration (e.g., Black-Scholes/Heston model fitting) across multiple cores.
5. **Result**: The result is returned to the Celery worker, which updates the database or triggers a downstream webhook.

## Key Files
- **Ray Workers**: `src/workers/ray_workers.py`.
- **Math Worker**: `src/workers/math_worker.py`.
- **Webhook Worker**: `src/workers/webhook_worker.py` (Implements circuit breakers for external service reliability).
