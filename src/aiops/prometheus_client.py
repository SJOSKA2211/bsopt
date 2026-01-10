from prometheus_api_client import PrometheusConnect
import structlog

logger = structlog.get_logger()

class PrometheusClient:
    """
    Wrapper for Prometheus API interactions.
    """
    def __init__(self, url: str):
        self.url = url
        self.prom = PrometheusConnect(url=self.url, disable_ssl=True)

    def check_connectivity(self):
        """
        Check if Prometheus is reachable.
        """
        try:
            # simple check: get all metrics names
            self.prom.all_metrics()
            logger.info("prometheus_connectivity_ok", url=self.url)
        except Exception as e:
            logger.error("prometheus_connectivity_failed", url=self.url, error=str(e))
            raise
