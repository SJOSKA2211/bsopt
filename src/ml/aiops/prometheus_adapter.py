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
            raise

    def get_5xx_error_rate(self, service: str) -> float:
        """Fetches the current 5xx error rate for a service."""
        if not service:
            raise ValueError("Service name cannot be empty")

        query = f'sum(rate(http_requests_total{{status=~"5..", service="{service}"}}[5m])) / sum(rate(http_requests_total{{service="{service}"}}[5m]))'
        try:
            result = self.prom.custom_query(query=query)
            if result:
                return float(result[0]["value"][1])
            return 0.0
        except Exception as e:
            logger.error("fetch_5xx_failed", service=service, error=str(e), query=query)
            return 0.0

    def get_p95_latency(self, service: str) -> float:
        """Fetches the 95th percentile latency for a service."""
        if not service:
            raise ValueError("Service name cannot be empty")

        query = f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{service="{service}"}}[5m])) by (le))'
        try:
            result = self.prom.custom_query(query=query)
            if result:
                return float(result[0]["value"][1])
            return 0.0
        except Exception as e:
            logger.error("fetch_p95_failed", service=service, error=str(e), query=query)
            return 0.0

    def get_metric_range(
        self, service: str, metric_name: str, duration: str = "1h", step: str = "1m"
    ) -> pd.DataFrame:
        """
        Fetches a range of metric values and returns a formatted DataFrame.
        Compatible with TFT model requirements.
        """
        logger.info(
            "fetching_metric_range",
            service=service,
            metric=metric_name,
            duration=duration,
        )

        from datetime import datetime

        end_time = datetime.now()
        start_time = end_time - self._parse_duration(duration)

        try:
            # Construct a query that targets the specific service and container
            query = f'sum(rate({metric_name}{{container="{service}"}}[5m]))'

            result = self.prom.custom_query_range(
                query=query,
                start_time=start_time,
                end_time=end_time,
                step=step,
            )

            if not result:
                logger.warning("prometheus_range_query_empty", query=query)
                return pd.DataFrame()

            # Parse Prometheus response format: [[timestamp, value], ...]
            data_points = result[0]["values"]
            df = pd.DataFrame(data_points, columns=["timestamp", "price"])
            df["price"] = df["price"].astype(float)
            df["symbol"] = service
            df["time_idx"] = np.arange(len(df))

            logger.info("metric_range_fetched_successfully", rows=len(df), service=service)
            return df
        except Exception as e:
            logger.error("metric_range_fetch_failed", error=str(e))
            return pd.DataFrame()

    def get_historical_metric_data(self, service: str, metric: str = "cpu_usage") -> np.ndarray:
        """Fetches univariate historical data for a metric."""
        df = self.get_metric_range(service, "container_cpu_usage_seconds_total")
        return df["price"].values if not df.empty else np.array([])

    def get_historical_metric_data_multi(self, service: str) -> np.ndarray:
        """Fetches multivariate historical data (CPU, Memory, Error Rate)."""
        # Returns a [samples, features] array
        try:
            cpu_df = self.get_metric_range(service, "container_cpu_usage_seconds_total")
            mem_df = self.get_metric_range(service, "container_memory_usage_bytes")

            if cpu_df.empty or mem_df.empty:
                return np.array([])

            # Properly align timestamps with bounds mapping to gracefully handle scrape skews
            import pandas as pd

            merged_df = pd.merge(
                cpu_df, mem_df, on="timestamp", how="outer", suffixes=("_cpu", "_mem")
            )
            merged_df = merged_df.sort_values("timestamp").ffill().bfill()

            return np.column_stack([merged_df["price_cpu"].values, merged_df["price_mem"].values])
        except Exception as e:
            logger.error("multivariate_data_fetch_failed", error=str(e))
            return np.array([])

    def _parse_duration(self, duration: str) -> timedelta:
        """Simple duration parser (e.g. 1h, 10m)."""
        unit = duration[-1]
        val = int(duration[:-1])
        if unit == "h":
            return timedelta(hours=val)
        if unit == "m":
            return timedelta(minutes=val)
        if unit == "s":
            return timedelta(seconds=val)
        return timedelta(minutes=10)

    async def get_latest_metrics_async(self, service: str = "api") -> pd.DataFrame:
        """Async-compatible wrapper for orchestrator."""
        import asyncio

        return await asyncio.to_thread(self.get_service_metrics, service)