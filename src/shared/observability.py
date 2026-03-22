import gc
import logging
import os
import random
import re
import time
import uuid
from collections.abc import Callable, Mapping, MutableMapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, cast

import httpx
import orjson
import structlog
from cachetools import LRUCache
from fastapi import Request, Response
from prometheus_client import (
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    push_to_gateway,
)

from src.shared.config import settings
from src.shared.off_heap_logger import omega_logger

# Thread pool for non-blocking telemetry and metric pushes
_METRICS_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="telemetry_background")

# Pre-compiled patterns for IP and Email
_IP_PATTERN = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.[\d]{1,3}\b")
_EMAIL_PATTERN = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b")

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


def _off_heap_processor(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> Mapping[str, Any] | str | bytes | bytearray | tuple[Any, ...]:
    """Zero-latency redirect for high-frequency logs."""
    if event_dict.get("high_frequency") or event_dict.get("latency_sensitive"):
        # Remove the marker before logging to SHM
        event_dict.pop("high_frequency", None)
        event_dict.pop("latency_sensitive", None)
        # Write to off-heap ring buffer
        omega_logger.log(event_dict.pop("event", "unknown"), **event_dict)
        # Prevent further processing by standard loggers
        raise structlog.DropEvent
    return event_dict


def _pii_masking_processor(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> Mapping[str, Any] | str | bytes | bytearray | tuple[Any, ...]:
    """Masks PII (IPs, Emails) in all log events for security compliance (Optimized)."""
    for key, value in event_dict.items():
        if isinstance(value, str):
            # Check if likely to contain PII before sub
            if "@" in value or "." in value:
                # Mask Email
                value = _EMAIL_PATTERN.sub("masked@email.com", value)
                # Mask IP
                value = _IP_PATTERN.sub("XXX.XXX.XXX.XXX", value)
                event_dict[key] = value
        elif key == "client_ip" and isinstance(value, str):
            # Specific handling for the client_ip key if it's already extracted
            if "." in value:
                parts = value.split(".")
                if len(parts) == 4:
                    event_dict[key] = f"{parts[0]}.{parts[1]}.{parts[2]}.xxx"

    return event_dict


def setup_logging() -> None:
    """Configures structlog for JSON logging (Loki compliant) with optimized processors."""
    structlog.configure(
        processors=[
            _TIME_STAMPER,
            _LEVEL_ADDER,
            _CALLSITE_ADDER,
            _pii_masking_processor,  # Global PII masking
            _off_heap_processor,  #  Redirect high-freq logs
            _JSON_RENDERER,
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.BoundLogger,
        cache_logger_on_first_use=True,
    )


def tune_gc(mode: str = "analytical") -> None:
    """
    Optimizes Garbage Collection based on the specific workload mode.
    - 'analytical': Standard aggressive collection to save memory.
    - 'high_frequency': Defer collection to avoid latency spikes during bursts.
    """
    if mode == "high_frequency":
        # Ultra-high thresholds for trading/streaming paths
        gc.set_threshold(100000, 500, 500)
    else:
        # Standard balanced tuning for API gateway/src
        gc.set_threshold(50000, 10, 10)

    structlog.get_logger().info("gc_tuned", mode=mode, thresholds=gc.get_threshold())


def tune_worker_resources() -> None:
    """
    OPTIMIZED: Coordinates CPU resource allocation for multi-backend parallelism.
    Prevents CPU oversubscription between Ray and Numba.
    """
    import os

    cpu_count = os.cpu_count() or 1
    # Assign 50% of cores to Numba to leave room for Ray/Event Loop
    numba_threads = max(1, cpu_count // 2)
    os.environ["NUMBA_NUM_THREADS"] = str(numba_threads)
    os.environ["MKL_NUM_THREADS"] = (
        "1"  # Force MKL to single thread to avoid nested parallel conflicts
    )
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    structlog.get_logger().info(
        "worker_resources_tuned", cpu_count=cpu_count, numba_threads=numba_threads
    )


_IP_CACHE = LRUCache(maxsize=1000)


async def logging_middleware(request: Request, call_next: Callable[[Request], Any]) -> Response:
    """FastAPI middleware for structured logging of every request with optimized IP masking and tracing."""
    logger = structlog.get_logger("api_request")
    start_time = time.time()

    # Trace Injection: Use existing ID or generate new one
    request_id = (
        request.headers.get("X-Correlation-ID")
        or request.headers.get("X-Request-ID")
        or str(uuid.uuid4())
    )
    request.state.request_id = request_id

    response = await call_next(request)

    duration = time.time() - start_time

    # IP masking is now handled globally by _pii_masking_processor
    client_ip = request.client.host if request.client else "unknown"

    # Log sampling: only log 10% of successful (2xx) requests to reduce I/O overhead.
    # Always log errors (4xx, 5xx) and redirects (3xx).
    should_log = True
    if 200 <= response.status_code < 300:
        if random.random() > getattr(settings, "LOG_SAMPLING_RATE", 0.1):  # nosec B311
            should_log = False

    if should_log:
        logger.info(
            "request_processed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration * 1000, 2),
            client_ip=client_ip,
            sampled=True if 200 <= response.status_code < 300 else False,
        )

    # Propagate ID back to client
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = str(round(duration * 1000, 2))
    return cast(Response, response)


# System Metrics (Defined at module level to avoid registration leaks)
PROCESS_CPU_USAGE = Gauge(
    "process_cpu_usage_percent", "CPU usage of the current process", ["service"]
)
PROCESS_MEMORY_USAGE = Gauge(
    "process_memory_usage_bytes", "RSS memory usage of the current process", ["service"]
)


_PROCESS_CACHE: Any = None


# System Metrics
def update_system_metrics(service_name: str) -> None:
    """Capture real-time resource utilization for the current process (Optimized)."""
    try:
        import psutil

        global _PROCESS_CACHE
        if _PROCESS_CACHE is None:
            _PROCESS_CACHE = psutil.Process()

        PROCESS_CPU_USAGE.labels(service=service_name).set(
            _PROCESS_CACHE.cpu_percent(interval=None)
        )
        PROCESS_MEMORY_USAGE.labels(service=service_name).set(_PROCESS_CACHE.memory_info().rss)
    except Exception:
        pass


def start_system_metrics_loop(service_name: str, interval: int = 15) -> None:
    """Starts a background thread to periodically update system metrics."""

    def _loop() -> None:
        while True:
            update_system_metrics(service_name)
            time.sleep(interval)

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"metrics_{service_name}")
    executor.submit(_loop)


# Common Metrics
SCRAPE_DURATION = Histogram(
    "market_scrape_duration_seconds",
    "Time spent scraping market data",
    ["api"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)
SCRAPE_ERRORS = Counter(
    "market_scrape_errors_total",
    "Total number of scrape errors",
    ["api", "status_code"],
)
TRAINING_DURATION = Histogram(
    "ml_training_duration_seconds", "Time spent in training", ["framework"]
)
MODEL_ACCURACY = Gauge(
    "ml_model_accuracy_score", "Accuracy score of the latest model", ["framework"]
)
MODEL_RMSE = Gauge("ml_model_rmse", "Root Mean Squared Error of model", ["model_type", "dataset"])
DATA_DRIFT_SCORE = Gauge("ml_data_drift_score", "PSI score for data drift")
MMD_DRIFT_SCORE = Gauge("ml_mmd_drift_score", "MMD score for multivariate data drift")
KS_TEST_SCORE = Gauge("ml_ks_test_p_value", "P-value from Kolmogorov-Smirnov test")
PERFORMANCE_DRIFT_ALERT = Gauge("ml_performance_drift_alert", "Binary alert for performance drift")
TRAINING_ERRORS = Counter("ml_training_errors_total", "Total training failures", ["framework"])

# Blockchain Metrics
BLOCKCHAIN_RPC_LATENCY = Histogram(
    "blockchain_rpc_latency_seconds",
    "Latency of RPC calls",
    ["method"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
BLOCKCHAIN_RPC_ERRORS = Counter(
    "blockchain_rpc_errors_total", "Total number of RPC errors", ["method"]
)
BLOCKCHAIN_GAS_PRICE = Gauge("blockchain_gas_price_gwei", "Current network gas price")

# Proxy/Scraper Metrics
PROXY_LATENCY = Histogram(
    "proxy_latency_seconds",
    "Latency of requests per proxy",
    ["proxy_url"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
PROXY_FAILURES = Counter("proxy_failures_total", "Total failures per proxy", ["proxy_url"])

# RL Agent Metrics
RL_EPISODE_REWARD = Gauge("rl_episode_reward_total", "Total reward per episode", ["agent_id"])
RL_ACTION_VARANCE = Gauge(
    "rl_action_variance", "Variance of actions taken by the RL agent", ["agent_id"]
)
RL_PORTFOLIO_VALUE = Gauge(
    "rl_portfolio_value_current",
    "Current portfolio value tracked by RL agent",
    ["agent_id"],
)


# Heston Metrics
HESTON_FELLER_MARGIN = Gauge(
    "heston_feller_margin", "Margin above Feller condition (2κθ - σ²)", ["symbol"]
)
CALIBRATION_DURATION = Histogram(
    "calibration_duration_seconds",
    "Time spent in calibration",
    ["symbol"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
HESTON_R_SQUARED = Gauge(
    "heston_r_squared",
    "R-squared coefficient of determination for Heston fit",
    ["symbol"],
)
HESTON_PARAMS_FRESHNESS = Gauge(
    "heston_params_freshness_seconds",
    "Time since last successful calibration",
    ["symbol"],
)

# ONNX & Pricing Service Metrics
ONNX_INFERENCE_LATENCY = Histogram(
    "onnx_inference_latency_ms",
    "Latency of ONNX inference in milliseconds",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0],
)
PRICING_SERVICE_DURATION = Histogram(
    "pricing_service_duration_seconds",
    "Time spent in PricingService methods",
    ["method"],
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0],
)
ML_PROXY_PREDICT_LATENCY = Histogram(
    "ml_proxy_predict_latency_seconds",
    "Latency of ML model predictions via proxy",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)


def observe_latency(histogram: Histogram, value: float, labels: dict[str, str] | None = None):
    """
    Non-blocking histogram observation.
    Dispatches to a background thread to prevent telemetry impact on pricing hot-loops.
    """
    if labels:
        _METRICS_EXECUTOR.submit(histogram.labels(**labels).observe, value)
    else:
        _METRICS_EXECUTOR.submit(histogram.observe, value)


def increment_counter(counter: Counter, amount: float = 1.0, labels: dict[str, str] | None = None):
    """
    Non-blocking counter increment.
    """
    if labels:
        _METRICS_EXECUTOR.submit(counter.labels(**labels).inc, amount)
    else:
        _METRICS_EXECUTOR.submit(counter.inc, amount)


def set_gauge(gauge: Gauge, value: float, labels: dict[str, str] | None = None):
    """
    Non-blocking gauge update.
    """
    if labels:
        _METRICS_EXECUTOR.submit(gauge.labels(**labels).set, value)
    else:
        _METRICS_EXECUTOR.submit(gauge.set, value)


def push_metrics(job_name: str) -> None:
    """
    Pushes all metrics to the Prometheus Pushgateway.
    Optimized: Dispatches to a background thread pool to avoid blocking the hot path.
    """
    gateway_url = os.environ.get("PUSHGATEWAY_URL")
    if not gateway_url:
        return

    def _do_push() -> None:
        try:
            push_to_gateway(gateway_url, job=job_name, registry=REGISTRY)
            # Use standard logging for background threads to avoid structlog contention
            logging.debug(f"metrics_pushed: {job_name}")
        except Exception as e:
            logging.error(f"metrics_push_failed: {e}")

    # Dispatch to background thread immediately
    _METRICS_EXECUTOR.submit(_do_push)


# Persistent HTTP client for observability
_observability_client: httpx.AsyncClient | None = None


def get_obs_client() -> httpx.AsyncClient:
    global _observability_client
    if _observability_client is None:
        _observability_client = httpx.AsyncClient(
            timeout=5.0,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
    return _observability_client


async def post_grafana_annotation(message: str, tags: list[str] | None = None) -> bool:
    """
    Posts an annotation to Grafana using a shared persistent client and high-speed serialization.
    """
    grafana_url = os.environ.get("GRAFANA_URL")
    if not grafana_url:
        structlog.get_logger().debug("grafana_annotation_skipped", reason="GRAFANA_URL not set")
        return False

    if tags is None:
        tags = []

    timestamp_ms = int(datetime.now(UTC).timestamp() * 1000)
    payload = {"time": timestamp_ms, "text": message, "tags": tags}

    client = get_obs_client()
    try:
        # Use orjson for faster serialization than the default json.dumps
        response = await client.post(
            f"{grafana_url}/api/annotations",
            headers={"Content-Type": "application/json"},
            content=orjson.dumps(payload),
        )
        response.raise_for_status()
        structlog.get_logger().info("grafana_annotation_posted", status_code=response.status_code)
        return True
    except Exception as e:
        structlog.get_logger().error("grafana_annotation_failed", error=str(e))
        return False
