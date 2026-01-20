import structlog
import os
import time
from prometheus_client import Summary, Counter, Gauge, Histogram, push_to_gateway, REGISTRY
from typing import List, Callable
from fastapi import Request, Response
import httpx
from datetime import datetime, timezone


def setup_logging():
    """Configures structlog for JSON logging (Loki compliant)."""
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.BoundLogger,
        cache_logger_on_first_use=True,
    )

async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """FastAPI middleware for structured logging of every request."""
    logger = structlog.get_logger("api_request")
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    # Mask client IP to protect PII
    client_ip = request.client.host if request.client else "unknown"
    if client_ip != "unknown":
        parts = client_ip.split('.')
        if len(parts) == 4:
            client_ip = f"{parts[0]}.{parts[1]}.{parts[2]}.xxx"
        else:
            client_ip = "masked"

    logger.info(
        "request_processed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration * 1000, 2),
        client_ip=client_ip
    )
    return response

# Common Metrics
SCRAPE_DURATION = Summary('market_scrape_duration_seconds', 'Time spent scraping market data', ['api'])
SCRAPE_ERRORS = Counter('market_scrape_errors_total', 'Total number of scrape errors', ['api', 'status_code'])
TRAINING_DURATION = Histogram('ml_training_duration_seconds', 'Time spent in training', ['framework'])
MODEL_ACCURACY = Gauge('ml_model_accuracy_score', 'Accuracy score of the latest model', ['framework'])
MODEL_RMSE = Gauge('ml_model_rmse', 'Root Mean Squared Error of model', ['model_type', 'dataset'])
DATA_DRIFT_SCORE = Gauge('ml_data_drift_score', 'PSI score for data drift')
KS_TEST_SCORE = Gauge('ml_ks_test_p_value', 'P-value from Kolmogorov-Smirnov test')
PERFORMANCE_DRIFT_ALERT = Gauge('ml_performance_drift_alert', 'Binary alert for performance drift')
TRAINING_ERRORS = Counter('ml_training_errors_total', 'Total training failures', ['framework'])

# Heston Metrics
HESTON_FELLER_MARGIN = Gauge('heston_feller_margin', 'Margin above Feller condition (2κθ - σ²)', ['symbol'])
CALIBRATION_DURATION = Histogram('calibration_duration_seconds', 'Time spent in calibration', ['symbol'])
HESTON_R_SQUARED = Gauge('heston_r_squared', 'R-squared coefficient of determination for Heston fit', ['symbol'])
HESTON_PARAMS_FRESHNESS = Gauge('heston_params_freshness_seconds', 'Time since last successful calibration', ['symbol'])

def push_metrics(job_name: str):
    """Pushes all metrics from the global REGISTRY to the Prometheus Pushgateway."""
    gateway_url = os.environ.get("PUSHGATEWAY_URL")
    if gateway_url:
        try:
            push_to_gateway(gateway_url, job=job_name, registry=REGISTRY)
            structlog.get_logger().info("metrics_pushed", job=job_name, gateway=gateway_url)
        except Exception as e:
            structlog.get_logger().error("metrics_push_failed", error=str(e))
    else:
        structlog.get_logger().debug("metrics_push_skipped", reason="no_gateway_url")


def post_grafana_annotation(message: str, tags: List[str] = None) -> bool:
    """
    Posts an annotation to Grafana.
    """
    grafana_url = os.environ.get("GRAFANA_URL")
    if not grafana_url:
        structlog.get_logger().debug("grafana_annotation_skipped", reason="GRAFANA_URL not set")
        return False

    if tags is None:
        tags = []

    # Grafana expects time in milliseconds Unix epoch
    timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    payload = {
        "time": timestamp_ms,
        "text": message,
        "tags": tags
    }

    try:
        # Assuming Grafana API endpoint is /api/annotations
        response = httpx.post(f"{grafana_url}/api/annotations", headers={"Content-Type": "application/json"}, json=payload, timeout=5)
        response.raise_for_status() # Raise an exception for bad status codes
        structlog.get_logger().info("grafana_annotation_posted", status_code=response.status_code, response=response.json())
        return True
    except httpx.HTTPStatusError as e:
        structlog.get_logger().error("grafana_annotation_failed", reason="HTTPStatusError", error=str(e), response_text=e.response.text)
        return False
    except httpx.RequestError as e:
        structlog.get_logger().error("grafana_annotation_failed", reason="RequestError", error=str(e))
        return False
    except Exception as e:
        structlog.get_logger().error("grafana_annotation_failed", reason="UnknownError", error=str(e))
        return False
