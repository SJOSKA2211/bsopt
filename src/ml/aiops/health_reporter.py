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
from src.shared.rabbitmq import get_rabbitmq
from src.database import get_async_engine
from sqlalchemy import text
from src.ml.aiops.remediators import RemediationPlanner
from src.ml.aiops.schemas import (
    MLHealthReport, 
    MLflowStatus, 
    PrometheusMetrics, 
    RedisAnomaly, 
    RabbitMQStatus, 
    RedisStatus,
    PostgresStatus,
    APIStatus,
    RemediationStatus,
    GuardianStatus
)

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
        self.rmq = get_rabbitmq()

    async def get_health_report(
        self, 
        planner: Any | None = None, 
        guardian: Any | None = None
    ) -> MLHealthReport:
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
        
        # 4. Fetch RabbitMQ status
        rabbitmq_status = await self._get_rabbitmq_status()

        # 5. Fetch Redis connectivity status
        redis_status = await self._get_redis_status()

        # 6. Fetch Postgres health status
        postgres_status = await self._get_postgres_status()
        
        # 7. Fetch API detailed status
        api_status = await self._get_api_status(prometheus_metrics)
        
        # 8. Fetch Remediation and Guardian statuses
        remediations = self._get_remediation_statuses(planner)
        guardian_status = self._get_guardian_status(guardian)
        
        # 8. Determine overall status
        status = "healthy"
        if mlflow_status.drift_detected or api_status.error_rate_5xx > 0.05 or \
           not rabbitmq_status.connected or not redis_status.connected or \
           not postgres_status.connected or not api_status.reachable:
            status = "degraded"
        if prometheus_metrics.error_rate_5xx > 0.2:
            status = "critical"

        return MLHealthReport(
            status=status,
            mlflow=mlflow_status,
            prometheus=prometheus_metrics,
            redis_anomalies=redis_anomalies,
            rabbitmq=rabbitmq_status,
            redis=redis_status,
            postgres=postgres_status,
            api=api_status,
            remediations=remediations,
            guardian=guardian_status,
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
            
            # Fetch request count (Sum of rates)
            req_count_query = f'sum(rate(http_requests_total{{service="{self.api_service_name}"}}[5m]))'
            req_count_res = await asyncio.to_thread(self.prometheus_client.prom.custom_query, query=req_count_query)
            request_count = 0
            if req_count_res:
                request_count = int(float(req_count_res[0]["value"][1]))

            return PrometheusMetrics(
                error_rate_5xx=error_rate,
                p95_latency=latency,
                cpu_usage=cpu_usage,
                memory_usage=mem_usage,
                request_count=request_count
            )
        except Exception as e:
            logger.error("prometheus_metrics_fetch_failed", error=str(e))
            return PrometheusMetrics(0.0, 0.0, 0.0, 0.0, 0)

    async def _get_api_status(self, metrics: PrometheusMetrics) -> APIStatus:
        """Determines if the API is reachable and healthy based on metrics."""
        # Simple reachability check (mock-friendly)
        reachable = True
        if metrics.request_count == 0 and metrics.error_rate_5xx == 0:
            reachable = False
            
        return APIStatus(
            reachable=reachable,
            p95_latency=round(metrics.p95_latency, 4),
            error_rate_5xx=round(metrics.error_rate_5xx, 4),
            request_count=metrics.request_count
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
    async def _get_rabbitmq_status(self) -> RabbitMQStatus:
        """Checks RabbitMQ connectivity and basic queue stats."""
        try:
            if not self.rmq.connection or self.rmq.connection.is_closed:
                await self.rmq.connect()
            
            connected = not self.rmq.connection.is_closed
            queue_depths = {}
            consumer_counts = {}

            # Check core system queues
            queues_to_check = [self.rmq.queue_name, self.rmq.audit_queue, self.rmq.dlq_name]
            for q_name in queues_to_check:
                try:
                    q = await self.rmq.channel.get_queue(q_name)
                    queue_depths[q_name] = q.declaration_result.message_count
                    consumer_counts[q_name] = q.declaration_result.consumer_count
                except Exception:
                    queue_depths[q_name] = -1
                    consumer_counts[q_name] = 0

            return RabbitMQStatus(
                connected=connected,
                queue_depths=queue_depths,
                consumer_counts=consumer_counts
            )
        except Exception as e:
            logger.error("rabbitmq_status_fetch_failed", error=str(e))
            return RabbitMQStatus(connected=False, queue_depths={}, consumer_counts={})

    async def _get_redis_status(self) -> RedisStatus:
        """Checks Redis connectivity and basic stats."""
        try:
            redis = await get_redis_client()
            info = await redis.info()
            
            # Extract basic metrics
            memory_usage = int(info.get("used_memory", 0))
            # dbsize() is typically preferred for key count if info isn't enough
            key_count = await redis.dbsize()
            
            return RedisStatus(
                connected=True,
                memory_usage_bytes=memory_usage,
                total_keys=key_count
            )
        except Exception as e:
            logger.error("redis_status_fetch_failed", error=str(e))
            return RedisStatus(connected=False, memory_usage_bytes=0, total_keys=0)

    async def _get_postgres_status(self) -> PostgresStatus:
        """Checks Postgres/TimescaleDB connectivity and detailed stats."""
        try:
            engine = get_async_engine()
            async with engine.connect() as conn:
                # 1. Version and basic stats
                res = await conn.execute(text("SELECT version(), count(*) FROM pg_stat_activity"))
                row = res.fetchone()
                version = str(row[0]) if row else "unknown"
                active_connections = int(row[1]) if row else 0
                
                # 2. Hypertable count
                hypertables = await conn.execute(
                    text("SELECT count(*) FROM timescaledb_information.hypertables")
                )
                hyper_count = hypertables.scalar() or 0
                
                # 3. Job count
                jobs = await conn.execute(
                    text("SELECT count(*) FROM timescaledb_information.jobs")
                )
                job_count = jobs.scalar() or 0
                
                # 4. Compression ratio (Simplified aggregate)
                compression = await conn.execute(
                    text("""
                    SELECT 
                        COALESCE(SUM(uncompressed_total_bytes) / NULLIF(SUM(compressed_total_bytes), 0), 1.0)
                    FROM timescaledb_information.compression_settings
                    JOIN timescaledb_information.hypertables ON hypertable_name = table_name
                """)
                )
                ratio = float(compression.scalar() or 1.0)
                
                return PostgresStatus(
                    connected=True,
                    version=version.split(",")[0],  # Short version string
                    active_connections=active_connections,
                    hypertables=int(hyper_count),
                    compression_ratio=round(ratio, 2),
                    job_count=int(job_count)
                )
        except Exception as e:
            logger.error("postgres_status_fetch_failed", error=str(e))
            return PostgresStatus(
                connected=False, version="unknown", active_connections=0, 
                hypertables=0, compression_ratio=1.0, job_count=0
            )

    def _get_remediation_statuses(self, planner: Any | None) -> List[RemediationStatus]:
        """Collects status info from all registered remediators."""
        if not planner:
            return []
        
        statuses = []
        for r in planner.remediators.values():
            statuses.append(RemediationStatus(
                name=r.name,
                status=r.get_status(),
                last_run=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(r.last_run)) if r.last_run > 0 else "never"
            ))
        return statuses

    def _get_guardian_status(self, guardian: Any | None) -> GuardianStatus:
        """Collects status from the Autonomous Guardian."""
        if not guardian:
            return GuardianStatus(active=False, safe_mode=False, paused_features=[])
        
        return GuardianStatus(
            active=guardian.is_active,
            safe_mode=getattr(guardian, "is_safe_mode", False),
            paused_features=getattr(guardian, "paused_features", [])
        )
