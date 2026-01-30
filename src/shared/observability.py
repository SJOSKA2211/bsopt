import structlog
import os
import time
import gc
from prometheus_client import Summary, Counter, Gauge, Histogram, push_to_gateway, REGISTRY
import logging
from typing import Any, Dict, List, Optional, Callable

from fastapi import Request, Response
import httpx
from datetime import datetime, timezone


# Pre-instantiate processors for performance
_TIME_STAMPER = structlog.processors.TimeStamper(fmt="iso")
_JSON_RENDERER = structlog.processors.JSONRenderer()
_LEVEL_ADDER = structlog.processors.add_log_level
_CALLSITE_ADDER = structlog.processors.CallsiteParameterAdder(
    {
        structlog.processors.CallsiteParameter.FILENAME,
        structlog.processors.CallsiteParameter.FUNC_NAME,
        structlog.processors.CallsiteParameter.LINENO,
    }
)

def setup_logging():
    """Configures structlog for JSON logging (Loki compliant) with optimized processors."""
    structlog.configure(
        processors=[
            _TIME_STAMPER,
            _LEVEL_ADDER,
            _CALLSITE_ADDER,
            _JSON_RENDERER,
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.BoundLogger,
        cache_logger_on_first_use=True,
    )

def tune_gc():
    """
    Optimizes Garbage Collection for long-running workers.
    Reduces latency spikes by increasing collection thresholds.
    """
    gc.set_threshold(50000, 10, 10)
    structlog.get_logger().info("gc_tuned", thresholds=gc.get_threshold())

async def logging_middleware(request: Request, call_next: Callable) -> Response:
# ... (rest of middleware)
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

# ONNX & Pricing Service Metrics
ONNX_INFERENCE_LATENCY = Histogram('onnx_inference_latency_ms', 'Latency of ONNX inference in milliseconds')
PRICING_SERVICE_DURATION = Histogram('pricing_service_duration_seconds', 'Time spent in PricingService methods', ['method'])
ML_PROXY_PREDICT_LATENCY = Histogram('ml_proxy_predict_latency_seconds', 'Latency of ML model predictions via proxy')

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


# Persistent HTTP client for observability
_observability_client: Optional[httpx.AsyncClient] = None

def get_obs_client() -> httpx.AsyncClient:
    global _observability_client
    if _observability_client is None:
        _observability_client = httpx.AsyncClient(
            timeout=5.0,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
    return _observability_client

async def post_grafana_annotation(message: str, tags: List[str] = None) -> bool:
    """
    Posts an annotation to Grafana using a shared persistent client.
    """
    grafana_url = os.environ.get("GRAFANA_URL")
    if not grafana_url:
        structlog.get_logger().debug("grafana_annotation_skipped", reason="GRAFANA_URL not set")
        return False

    if tags is None:
        tags = []

    timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    payload = {"time": timestamp_ms, "text": message, "tags": tags}

    client = get_obs_client()
    try:
        response = await client.post(
            f"{grafana_url}/api/annotations", 
            headers={"Content-Type": "application/json"}, 
            json=payload
        )
        response.raise_for_status()
        structlog.get_logger().info("grafana_annotation_posted", status_code=response.status_code)
        return True
    except Exception as e:
        structlog.get_logger().error("grafana_annotation_failed", error=str(e))
        return False
