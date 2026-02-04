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


from src.shared.off_heap_logger import omega_logger

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

def _off_heap_processor(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """🚀 SINGULARITY: Zero-latency redirect for high-frequency logs."""
    if event_dict.get("high_frequency") or event_dict.get("latency_sensitive"):
        # Remove the marker before logging to SHM
        event_dict.pop("high_frequency", None)
        event_dict.pop("latency_sensitive", None)
        # Write to off-heap ring buffer
        omega_logger.log(event_dict.pop("event", "unknown"), **event_dict)
        # Prevent further processing by standard loggers
        raise structlog.DropEvent
    return event_dict

def setup_logging():
    """Configures structlog for JSON logging (Loki compliant) with optimized processors."""
    structlog.configure(
        processors=[
            _TIME_STAMPER,
            _LEVEL_ADDER,
            _CALLSITE_ADDER,
            _off_heap_processor, # 🚀 Redirect high-freq logs
            _JSON_RENDERER,
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.BoundLogger,
        cache_logger_on_first_use=True,
    )

def tune_gc(mode: str = "analytical"):
    """
    Optimizes Garbage Collection based on the specific workload mode.
    - 'analytical': Standard aggressive collection to save memory.
    - 'high_frequency': Defer collection to avoid latency spikes during bursts.
    """
    if mode == "high_frequency":
        # Ultra-high thresholds for trading/streaming paths
        gc.set_threshold(100000, 500, 500)
    else:
        # Standard balanced tuning for API gateway/services
        gc.set_threshold(50000, 10, 10)
        
    structlog.get_logger().info("gc_tuned", mode=mode, thresholds=gc.get_threshold())

def tune_worker_resources():
    """
    SOTA: Coordinates CPU resource allocation for multi-backend parallelism.
    Prevents CPU oversubscription between Ray and Numba.
    """
    import os
    import psutil
    
    cpu_count = os.cpu_count() or 1
    # Assign 50% of cores to Numba to leave room for Ray/Event Loop
    numba_threads = max(1, cpu_count // 2)
    os.environ["NUMBA_NUM_THREADS"] = str(numba_threads)
    os.environ["MKL_NUM_THREADS"] = "1" # Force MKL to single thread to avoid nested parallel conflicts
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    structlog.get_logger().info(
        "worker_resources_tuned", 
        cpu_count=cpu_count, 
        numba_threads=numba_threads
    )

import orjson
from cachetools import LRUCache

_IP_CACHE = LRUCache(maxsize=1000)

async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """FastAPI middleware for structured logging of every request with optimized IP masking."""
    logger = structlog.get_logger("api_request")
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    # Mask client IP to protect PII with caching for performance
    raw_ip = request.client.host if request.client else "unknown"
    if raw_ip in _IP_CACHE:
        client_ip = _IP_CACHE[raw_ip]
    elif raw_ip != "unknown":
        parts = raw_ip.split('.')
        if len(parts) == 4:
            client_ip = f"{parts[0]}.{parts[1]}.{parts[2]}.xxx"
        else:
            client_ip = "masked"
        _IP_CACHE[raw_ip] = client_ip
    else:
        client_ip = "unknown"

    # Log sampling: only log 10% of successful (2xx) requests to reduce I/O overhead.
    # Always log errors (4xx, 5xx) and redirects (3xx).
    import random
    should_log = True
    if 200 <= response.status_code < 300:
        if random.random() > 0.1: # 10% sampling rate
            should_log = False

    if should_log:
        logger.info(
            "request_processed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration * 1000, 2),
            client_ip=client_ip,
            sampled=True if 200 <= response.status_code < 300 else False
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

# Blockchain Metrics
BLOCKCHAIN_RPC_LATENCY = Histogram('blockchain_rpc_latency_seconds', 'Latency of RPC calls', ['method'])
BLOCKCHAIN_RPC_ERRORS = Counter('blockchain_rpc_errors_total', 'Total number of RPC errors', ['method'])
BLOCKCHAIN_GAS_PRICE = Gauge('blockchain_gas_price_gwei', 'Current network gas price')

# Proxy/Scraper Metrics
PROXY_LATENCY = Histogram('proxy_latency_seconds', 'Latency of requests per proxy', ['proxy_url'])
PROXY_FAILURES = Counter('proxy_failures_total', 'Total failures per proxy', ['proxy_url'])

# RL Agent Metrics
RL_EPISODE_REWARD = Gauge('rl_episode_reward_total', 'Total reward per episode', ['agent_id'])
RL_ACTION_VARIANCE = Gauge('rl_action_variance', 'Variance of actions taken by the RL agent', ['agent_id'])
RL_PORTFOLIO_VALUE = Gauge('rl_portfolio_value_current', 'Current portfolio value tracked by RL agent', ['agent_id'])


# Heston Metrics
HESTON_FELLER_MARGIN = Gauge('heston_feller_margin', 'Margin above Feller condition (2κθ - σ²)', ['symbol'])
CALIBRATION_DURATION = Histogram('calibration_duration_seconds', 'Time spent in calibration', ['symbol'])
HESTON_R_SQUARED = Gauge('heston_r_squared', 'R-squared coefficient of determination for Heston fit', ['symbol'])
HESTON_PARAMS_FRESHNESS = Gauge('heston_params_freshness_seconds', 'Time since last successful calibration', ['symbol'])

# ONNX & Pricing Service Metrics
ONNX_INFERENCE_LATENCY = Histogram('onnx_inference_latency_ms', 'Latency of ONNX inference in milliseconds')
PRICING_SERVICE_DURATION = Histogram('pricing_service_duration_seconds', 'Time spent in PricingService methods', ['method'])
ML_PROXY_PREDICT_LATENCY = Histogram('ml_proxy_predict_latency_seconds', 'Latency of ML model predictions via proxy')

import asyncio
from concurrent.futures import ThreadPoolExecutor

# Pre-instantiate a dedicated thread pool for off-heap metrics ingestion
_METRICS_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="metrics_pusher")

def push_metrics(job_name: str):
    """
    Pushes all metrics to the Prometheus Pushgateway.
    Optimized: Dispatches to a background thread pool to avoid blocking the hot path.
    """
    gateway_url = os.environ.get("PUSHGATEWAY_URL")
    if not gateway_url:
        return

    def _do_push():
        try:
            push_to_gateway(gateway_url, job=job_name, registry=REGISTRY)
            # Use standard logging for background threads to avoid structlog contention
            logging.debug(f"metrics_pushed: {job_name}")
        except Exception as e:
            logging.error(f"metrics_push_failed: {e}")

    # Dispatch to background thread immediately
    _METRICS_EXECUTOR.submit(_do_push)


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
    Posts an annotation to Grafana using a shared persistent client and high-speed serialization.
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
        # Use orjson for faster serialization than the default json.dumps
        response = await client.post(
            f"{grafana_url}/api/annotations", 
            headers={"Content-Type": "application/json"}, 
            content=orjson.dumps(payload)
        )
        response.raise_for_status()
        structlog.get_logger().info("grafana_annotation_posted", status_code=response.status_code)
        return True
    except Exception as e:
        structlog.get_logger().error("grafana_annotation_failed", error=str(e))
        return False
