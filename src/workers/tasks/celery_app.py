import os

import orjson
import structlog
from celery import Celery
from celery.signals import worker_process_init
from kombu import Exchange, Queue
from kombu.serialization import register


def orjson_dumps(obj):
    return orjson.dumps(obj)

def orjson_loads(s):
    return orjson.loads(s)

register('orjson', orjson_dumps, orjson_loads,
         content_type='application/x-orjson',
         content_encoding='utf-8')

from src.shared.config import settings

logger = structlog.get_logger(__name__)

# ==============================================================================
# Manifold: UNIFIED CELERY MANIFOLD (v5.0)
# ==============================================================================

celery_app = Celery(
    "Manifold",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.REDIS_URL,
    include=[
        "src.ingestion.tasks",
        "src.ml.pipelines.retraining",
        "src.ml.aiops.remediators",
        "src.workers.tasks.ml_tasks",
        "src.workers.tasks.pricing_tasks",
        "src.workers.tasks.trading_tasks",
        "src.workers.tasks.security_tasks",
        "src.workers.tasks.data_tasks",
        "src.shared.tasks.audit_tasks",
    ],
)

# Robust Configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
    worker_send_task_events=True,
    task_send_sent_event=True,
    broker_connection_retry_on_startup=True,
)

# Queue Definition (Zero-Placeholder Isolation)
celery_app.conf.task_queues = (
    Queue("default", Exchange("default"), routing_key="default"),
    Queue("ingestion", Exchange("ingestion"), routing_key="ingestion"),
    Queue("mlops", Exchange("mlops"), routing_key="mlops"),
    Queue("high_priority", Exchange("high_priority"), routing_key="high_priority"),
)

celery_app.conf.task_default_queue = "default"
celery_app.conf.task_default_exchange = "default"
celery_app.conf.task_default_routing_key = "default"

# Task Routing
celery_app.conf.task_routes = {
    "scrapers.*": {"queue": "ingestion"},
    "ml.*": {"queue": "mlops"},
    "aiops.*": {"queue": "mlops"},
}


class BaseTaskWithRetry(celery_app.Task):
    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 5}
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True


@worker_process_init.connect
def init_worker(**kwargs):
    """Initialize resources for each worker process."""
    logger.info("Worker process initializing...", pid=os.getpid())
    # Database connections, ML models, etc. can be pre-loaded here.
