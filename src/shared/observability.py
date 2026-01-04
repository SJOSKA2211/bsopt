import structlog
from prometheus_client import Summary, Counter, Gauge

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
MODEL_ACCURACY = Gauge('ml_model_accuracy_score', 'Accuracy score of the latest model', ['framework'])
DATA_DRIFT_SCORE = Gauge('ml_data_drift_score', 'PSI score for data drift')
KS_TEST_SCORE = Gauge('ml_ks_test_p_value', 'P-value from Kolmogorov-Smirnov test')
PERFORMANCE_DRIFT_ALERT = Gauge('ml_performance_drift_alert', 'Binary alert for performance drift')
