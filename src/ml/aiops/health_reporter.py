import time
import asyncio
import structlog
from typing import List, Any, Dict, Optional
import msgspec

try:
    from mlflow.tracking import MlflowClient
except ImportError:
    MlflowClient = None

from src.ml.aiops.prometheus_adapter import PrometheusClient
from src.shared.utils.cache import get_redis_client
from src.ml.aiops.schemas import MLHealthReport, MLflowStatus, PrometheusMetrics, RedisAnomaly

logger = structlog.get_logger(__name__)

class HealthReporter:
    """
    Aggregator for MLflow, Prometheus, and Redis metrics.
    Provides a centralized health report for the ML Manifold.
    """

    def __init__(self, prometheus_url: str, api_service_name: str = "bsopt-api"):
        self.mlflow_client = MlflowClient() if MlflowClient else None
        self.prometheus_client = PrometheusClient(url=prometheus_url)
        self.api_service_name = api_service_name
        self.anomaly_history_key = "aiops:anomaly_history"

    async def get_health_report(self) -> MLHealthReport:
        """
        Aggregates metrics from all sources and returns a unified health report.
        """
        logger.info("generating_ml_health_report")
        
        # 1. Fetch MLflow status
        mlflow_status = self._get_mlflow_status()
        
        # 2. Fetch Prometheus metrics
        prometheus_metrics = await self._get_prometheus_metrics()
        
        # 3. Fetch Redis anomalies
        redis_anomalies = await self._get_redis_anomalies()
        
        # 4. Determine overall status
        status = "healthy"
        if mlflow_status.drift_detected or prometheus_metrics.error_rate_5xx > 0.05:
            status = "degraded"
        if prometheus_metrics.error_rate_5xx > 0.2:
            status = "critical"

        return MLHealthReport(
            status=status,
            mlflow=mlflow_status,
            prometheus=prometheus_metrics,
            redis_anomalies=redis_anomalies,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )

    def _get_mlflow_status(self) -> MLflowStatus:
        if not self.mlflow_client:
            return MLflowStatus(stage="unknown", drift_detected=False)
            
        try:
            # Get latest run from default experiment
            runs = self.mlflow_client.search_runs(
                experiment_ids=["0"],
                max_results=1,
                order_by=["attribute.start_time DESC"]
            )
            
            if not runs:
                return MLflowStatus(stage="unknown", drift_detected=False)
            
            latest_run = runs[0]
            drift_detected = latest_run.data.tags.get("drift_detected", "false").lower() == "true"
            stage = latest_run.data.tags.get("stage", "development")
            
            return MLflowStatus(
                stage=stage,
                drift_detected=drift_detected,
                last_run_id=latest_run.info.run_id
            )
        except Exception as e:
            logger.error("mlflow_status_fetch_failed", error=str(e))
            return MLflowStatus(stage="error", drift_detected=False)

    async def _get_prometheus_metrics(self) -> PrometheusMetrics:
        try:
            # Fetch metrics concurrently
            error_rate_task = asyncio.to_thread(
                self.prometheus_client.get_5xx_error_rate, self.api_service_name
            )
            latency_task = asyncio.to_thread(
                self.prometheus_client.get_p95_latency, self.api_service_name
            )
            
            # For CPU and Memory, we might need to query specific metrics
            cpu_query = f'sum(rate(container_cpu_usage_seconds_total{{container="{self.api_service_name}"}}[5m]))'
            mem_query = f'sum(container_memory_usage_bytes{{container="{self.api_service_name}"}})'
            
            cpu_task = asyncio.to_thread(self.prometheus_client.prom.custom_query, query=cpu_query)
            mem_task = asyncio.to_thread(self.prometheus_client.prom.custom_query, query=mem_query)
            
            results = await asyncio.gather(error_rate_task, latency_task, cpu_task, mem_task)
            
            error_rate = results[0]
            latency = results[1]
            
            cpu_usage = 0.0
            if results[2] and len(results[2]) > 0:
                cpu_usage = float(results[2][0]["value"][1])
                
            mem_usage = 0.0
            if results[3] and len(results[3]) > 0:
                mem_usage = float(results[3][0]["value"][1])
                
            return PrometheusMetrics(
                error_rate_5xx=error_rate,
                p95_latency=latency,
                cpu_usage=cpu_usage,
                memory_usage=mem_usage
            )
        except Exception as e:
            logger.error("prometheus_metrics_fetch_failed", error=str(e))
            return PrometheusMetrics(
                error_rate_5xx=0.0,
                p95_latency=0.0,
                cpu_usage=0.0,
                memory_usage=0.0
            )

    async def _get_redis_anomalies(self) -> List[RedisAnomaly]:
        try:
            redis = await get_redis_client()
            # Get last 10 anomalies from history
            raw_anomalies = await redis.lrange(self.anomaly_history_key, 0, 9)
            
            anomalies = []
            for raw in raw_anomalies:
                try:
                    data = msgspec.json.decode(raw)
                    anomalies.append(RedisAnomaly(
                        timestamp=data.get("timestamp", ""),
                        description=data.get("description", ""),
                        severity=data.get("severity", "medium")
                    ))
                except Exception as decode_err:
                    logger.error("failed_to_decode_anomaly", error=str(decode_err))
            return anomalies
        except Exception as e:
            logger.error("redis_anomalies_fetch_failed", error=str(e))
            return []
