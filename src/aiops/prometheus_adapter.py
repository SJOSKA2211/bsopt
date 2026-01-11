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

    def get_5xx_error_rate(self, service: str) -> float:
        """
        Fetch the 5xx error rate for a given service over the last 5 minutes.
        """
        if not service:
            raise ValueError("Service name cannot be empty")
            
        query = f'sum(rate(http_requests_total{{status=~"5..", service="{service}"}}[5m])) / sum(rate(http_requests_total{{service="{service}"}}[5m]))'
        try:
            result = self.prom.custom_query(query=query)
            if result and "value" in result[0]:
                return float(result[0]["value"][1])
            return 0.0
        except Exception as e:
            logger.error("fetch_5xx_failed", service=service, error=str(e), query=query)
            return 0.0

    def get_p95_latency(self, service: str) -> float:
        """
        Fetch the p95 latency for a given service over the last 5 minutes.
        """
        if not service:
            raise ValueError("Service name cannot be empty")

        query = f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{service="{service}"}}[5m])) by (le))'
        try:
            result = self.prom.custom_query(query=query)
            if result and "value" in result[0]:
                return float(result[0]["value"][1])
            return 0.0
        except Exception as e:
            logger.error("fetch_p95_failed", service=service, error=str(e), query=query)
            return 0.0
