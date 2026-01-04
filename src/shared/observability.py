import structlog
from prometheus_client import Summary, Counter, Gauge, Histogram

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
