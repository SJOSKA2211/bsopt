from celery import Celery
from datetime import datetime
import time

# Basic configuration will be loaded from environment variables via docker-compose
# These defaults are for local development if .env is not fully populated yet.

celery_app = Celery("bsopt_tasks")

# Configuration loaded from environment variables (e.g., CELERY_BROKER_URL, REDIS_URL)
# Example settings if not provided:
# celery_app.conf.broker_url = "amqp://guest:guest@rabbitmq:5672//"
# celery_app.conf.result_backend = "redis://:test_redis_password_v2@redis:6379/0"
# celery_app.conf.task_ignore_result = False
# celery_app.conf.task_track_started = True

@celery_app.task
def process_data_task(data: str):
    """A sample task to simulate background data processing."""
    timestamp = datetime.utcnow().isoformat()
    result = f"Processed: {data} at {timestamp}"
    print(result)
    time.sleep(2) # Simulate work
    return result

@celery_app.task
def trigger_ml_training_task(model_id: str, epochs: int, batch_size: int):
    """Simulates triggering an ML model training job."""
    timestamp = datetime.utcnow().isoformat()
    result = f"ML training triggered for model {model_id} (Epochs: {epochs}, Batch Size: {batch_size}) at {timestamp}"
    print(result)
    time.sleep(5) # Simulate longer training process
    return result

@celery_app.task
def simulate_market_data_ingestion(symbol: str, num_days: int):
    """Simulates ingesting historical market data."""
    timestamp = datetime.utcnow().isoformat()
    result = f"Simulating ingestion of {num_days} days of market data for {symbol} at {timestamp}"
    print(result)
    time.sleep(3) # Simulate work
    return result

# Add more tasks as needed for various background operations.
