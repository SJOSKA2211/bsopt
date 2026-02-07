import numpy as np
import pandas as pd
import structlog
from prometheus_api_client import PrometheusConnect
from datetime import datetime, timedelta

logger = structlog.get_logger()

class PrometheusAdapter:
    """
    Advanced adapter for Prometheus that fetches multivariate system metrics
    and returns them as Pandas DataFrames for anomaly and drift detection.
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

    def get_service_metrics(self, service: str, duration: str = "10m", step: str = "15s") -> pd.DataFrame:
        """
        Fetches a comprehensive set of metrics for a service as a DataFrame.
        Includes latency, error rates, CPU, and memory.
        """
        logger.info("fetching_service_metrics", service=service, duration=duration)
        
        # Define queries
        queries = {
            "p95_latency": f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{service="{service}"}}[5m])) by (le))',
            "error_rate": f'sum(rate(http_requests_total{{status=~"5..", service="{service}"}}[5m])) / sum(rate(http_requests_total{{service="{service}"}}[5m]))',
            "cpu_usage": f'sum(rate(container_cpu_usage_seconds_total{{container="{service}"}}[5m]))',
            "memory_usage": f'sum(container_memory_usage_bytes{{container="{service}"}})'
        }
        
        df_dict = {}
        
        for name, query in queries.items():
            try:
                # Use get_metric_range_data for timeseries
                end_time = datetime.now()
                start_time = end_time - self._parse_duration(duration)
                
                result = self.prom.get_metric_range_data(
                    metric_name=None, # Use custom query instead
                    label_config=None,
                    start_time=start_time,
                    end_time=end_time,
                    chunk_size=None,
                    store_locally=False,
                    custom_query=query
                )
                
                if result:
                    # result is a list of dicts, take the first one
                    values = result[0].get("values", [])
                    # values is [[timestamp, value], ...]
                    if values:
                        times = [v[0] for v in values]
                        vals = [float(v[1]) for v in values]
                        
                        # Create a series indexed by timestamp
                        # Round times to handle slight alignment issues
                        series = pd.Series(vals, index=np.round(times).astype(int), name=name)
                        df_dict[name] = series
            except Exception as e:
                logger.error("metric_fetch_failed", metric=name, error=str(e))
                # Fill with 0.0 or NaN to maintain DataFrame structure
                df_dict[name] = pd.Series(dtype=float)

        if not df_dict:
            return pd.DataFrame()

        # Join all series on timestamp
        df = pd.DataFrame(df_dict).fillna(method='ffill').fillna(0.0)
        df.index.name = "timestamp"
        
        logger.info("service_metrics_fetched", service=service, rows=len(df))
        return df

    def _parse_duration(self, duration: str) -> timedelta:
        """Simple duration parser (e.g. 1h, 10m)."""
        unit = duration[-1]
        val = int(duration[:-1])
        if unit == 'h': return timedelta(hours=val)
        if unit == 'm': return timedelta(minutes=val)
        if unit == 's': return timedelta(seconds=val)
        return timedelta(minutes=10)

    async def get_latest_metrics_async(self, service: str = "api") -> pd.DataFrame:
        """Async-compatible wrapper for orchestrator."""
        import asyncio
        return await asyncio.to_thread(self.get_service_metrics, service)
