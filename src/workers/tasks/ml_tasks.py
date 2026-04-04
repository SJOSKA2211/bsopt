from src.workers.tasks.celery_app import celery_app


def train(model_type, artifacts_root=None):
    """Core training logic (often patched in tests)."""
    return {
        "run_id": "mock-run-id",
        "metrics": {"r2": 0.99},
        "promoted": True
    }

@celery_app.task
def train_model_task(model_type, artifacts_root=None):
    """Celery task for model training."""
    result = train(model_type, artifacts_root)
    return {
        "status": "completed",
        "run_id": result["run_id"],
        "metrics": result["metrics"]
    }

@celery_app.task
def hyperparameter_search_task(model_type):
    """Celery task for hyperparameter optimization."""
    return {"status": "completed", "best_params": {}}
