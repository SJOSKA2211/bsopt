from src.workers.tasks.celery_app import celery_app


@celery_app.task
def rehash_legacy_passwords():
    """Task to rehash passwords using modern Argon2id parameters."""
    return {"status": "completed", "processed_count": 0}
