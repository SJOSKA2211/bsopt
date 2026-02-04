# Unify MLflow Tracking Implementation Plan

## Overview
Enable centralized, scalable experiment tracking by configuring MLflow to use the Neon (Postgres) backend for metadata and model storage.

## Current State Analysis
- **Tracking**: `src/ml/training/train_all.py:100` hardcodes a local `file://` URI.
- **Neon Backend**: Already used for primary data (`src/config.py`) but MLflow is not taking advantage of it.

## Implementation Approach
1. Refactor `train_all` to use `settings.DATABASE_URL` for the MLflow tracking URI.
2. Ensure the Postgres URL is correctly formatted for MLflow (removing any incompatible prefixes like `asyncpg`).
3. Standardize experiment naming across the pipeline.

## Phase 1: Neon Integration for MLflow
### Overview
Redirect MLflow tracking to the serverless database.

### Changes Required:

#### 1. src/ml/training/train_all.py
**Changes**: Update tracking URI to use global settings.
```python
    # 4. MLflow Tracking
    from src.config import get_settings
    settings = get_settings()
    
    # MLflow needs standard postgresql prefix (not asyncpg)
    tracking_uri = settings.DATABASE_URL.replace("postgresql+asyncpg", "postgresql")
    mlflow.set_tracking_uri(tracking_uri)
    
    logger.info("mlflow_tracking_redirected", target="neon")
```

### Success Criteria:
#### Automated:
- [ ] `pytest tests/test_config.py` passes.
#### Manual:
- [ ] Confirm new experiments appear in the Neon database (check `mlflow_experiments` table).

**Implementation Note**: Requires `psycopg2-binary` to be installed.
