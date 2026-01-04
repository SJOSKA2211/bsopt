import structlog
import os
from prometheus_client import Summary, Counter, Gauge, Histogram, push_to_gateway, REGISTRY

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

# Common Metrics
SCRAPE_DURATION = Summary('market_scrape_duration_seconds', 'Time spent scraping market data', ['api'])
SCRAPE_ERRORS = Counter('market_scrape_errors_total', 'Total number of scrape errors', ['api', 'status_code'])
TRAINING_DURATION = Summary('ml_training_duration_seconds', 'Time spent in training', ['framework'])
TRAINING_DURATION_HISTOGRAM = Histogram('ml_training_duration_hist_seconds', 'Time spent training model', ['framework'])
MODEL_ACCURACY = Gauge('ml_model_accuracy_score', 'Accuracy score of the latest model', ['framework'])
MODEL_RMSE = Gauge('ml_model_rmse', 'Root Mean Squared Error of model', ['model_type', 'dataset'])
DATA_DRIFT_SCORE = Gauge('ml_data_drift_score', 'PSI score for data drift')
KS_TEST_SCORE = Gauge('ml_ks_test_p_value', 'P-value from Kolmogorov-Smirnov test')
PERFORMANCE_DRIFT_ALERT = Gauge('ml_performance_drift_alert', 'Binary alert for performance drift')
TRAINING_ERRORS = Counter('ml_training_errors_total', 'Total training failures', ['framework'])

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
