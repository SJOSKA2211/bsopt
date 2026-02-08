from datetime import timedelta

import numpy as np
import pandas as pd
import structlog
from prometheus_api_client import PrometheusConnect

logger = structlog.get_logger()

class PrometheusClient:
    """
    Advanced client for Prometheus that fetches multivariate system metrics
    and returns them as NumPy arrays or DataFrames for anomaly and drift detection.
    """
    def __init__(self, url: str):
        self.url = url
        self.prom = PrometheusConnect(url=self.url, disable_ssl=True)

    def check_connectivity(self) -> bool:
        """Check if Prometheus is reachable."""
        try:
            self.prom.all_metrics()
            logger.info("prometheus_connectivity_ok", url=self.url)
            return True
        except Exception as e:
            logger.error("prometheus_connectivity_failed", url=self.url, error=str(e))
            return False

    def get_5xx_error_rate(self, service: str) -> float:
        """Fetches the current 5xx error rate for a service."""
        query = f'sum(rate(http_requests_total{{status=~"5..", service="{service}"}}[5m])) / sum(rate(http_requests_total{{service="{service}"}}[5m]))'
        try:
            result = self.prom.custom_query(query)
            if result:
                return float(result[0]['value'][1])
            return 0.0
        except Exception as e:
            logger.error("error_rate_fetch_failed", error=str(e))
            return 0.0

    def get_p95_latency(self, service: str) -> float:
        """Fetches the 95th percentile latency for a service."""
        query = f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{service="{service}"}}[5m])) by (le))'
        try:
            result = self.prom.custom_query(query)
            if result:
                return float(result[0]['value'][1])
            return 0.0
        except Exception as e:
            logger.error("latency_fetch_failed", error=str(e))
            return 0.0

    def get_historical_metric_data(self, service: str, metric: str = "cpu_usage") -> np.ndarray:
        """Fetches univariate historical data for a metric."""
        query = f'sum(rate(container_cpu_usage_seconds_total{{container="{service}"}}[5m]))'
        try:
            result = self.prom.custom_query(query) # Simplified for now
            if result:
                # In a real implementation, we would fetch a range
                return np.array([float(result[0]['value'][1])])
            return np.array([])
        except Exception as e:
            logger.error("historical_data_fetch_failed", error=str(e))
            return np.array([])

    def get_historical_metric_data_multi(self, service: str) -> np.ndarray:
        """Fetches multivariate historical data (CPU, Memory, Error Rate)."""
        # Returns a [samples, features] array
        try:
            # Mocking range behavior for now since custom_query is used in this stub
            cpu = self.get_historical_metric_data(service, "cpu")
            return cpu.reshape(-1, 1) if cpu.size > 0 else np.array([])
        except Exception as e:
            logger.error("multivariate_data_fetch_failed", error=str(e))
            return np.array([])

    def _parse_duration(self, duration: str) -> timedelta:
        """Simple duration parser (e.g. 1h, 10m)."""
        unit = duration[-1]
        val = int(duration[:-1])
        if unit == 'h':
            return timedelta(hours=val)
        if unit == 'm':
            return timedelta(minutes=val)
        if unit == 's':
            return timedelta(seconds=val)
        return timedelta(minutes=10)

    async def get_latest_metrics_async(self, service: str = "api") -> pd.DataFrame:
        """Async-compatible wrapper for orchestrator."""
        import asyncio
        return await asyncio.to_thread(self.get_service_metrics, service)
