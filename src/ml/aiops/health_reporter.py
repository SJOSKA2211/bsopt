import asyncio
import os
import time
from typing import Any

import msgspec

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
import structlog
from mlflow.tracking import MlflowClient
from sqlalchemy import text

from src.database import get_async_engine
from src.ml.aiops.prometheus_adapter import PrometheusClient
from src.ml.aiops.schemas import (
    APIStatus,
    AuthStatus,
    GuardianStatus,
    IngestionStatus,
    MathKernelStatus,
    MLflowStatus,
    MLHealthReport,
    MLInferenceStatus,
    PortfolioStatus,
    PostgresStatus,
    PrometheusMetrics,
    RabbitMQStatus,
    RayStatus,
    RedisAnomaly,
    RedisStatus,
    RemediationStatus,
    WorkerStatus,
)
from src.shared.rabbitmq import get_rabbitmq
from src.shared.utils.cache import get_redis_client

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
        self.auth_service_name = "auth-service"
        self.ingestion_service_name = "ingestion-service"
        self.portfolio_service_name = "portfolio-service"
        self.math_kernel_service_name = "math-kernel"
        self.ml_inference_service_name = "ml-inference"
        self.anomaly_history_key = "aiops:anomaly_history"
        self.rmq = get_rabbitmq()

    async def get_health_report(
        self, planner: Any | None = None, guardian: Any | None = None
    ) -> MLHealthReport:
        """
        Aggregates metrics from all sources and returns a unified health report.
        """
        logger.info("generating_ml_health_report")

        # Fetch all statuses concurrently
        tasks = [
            asyncio.to_thread(self._get_mlflow_status),
            self._get_prometheus_metrics(),
            self._get_redis_anomalies(),
            self._get_rabbitmq_status(),
            self._get_redis_status(),
            self._get_postgres_status(),
            self._get_auth_status(),
            self._get_ingestion_status(),
            self._get_portfolio_status(),
            self._get_math_kernel_status(),
            self._get_ml_inference_status(),
            self._get_worker_status(),
            self._get_ray_status(),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Extract results safely
        (
            mlflow_status,
            prometheus_metrics,
            redis_anomalies,
            rabbitmq_status,
            redis_status,
            postgres_status,
            auth_status,
            ingestion_status,
            portfolio_status,
            math_kernel_status,
            ml_inference_status,
            worker_status,
            ray_status,
        ) = [
            res if not isinstance(res, Exception) else self._get_default_status(i, res)
            for i, res in enumerate(results)
        ]

        # 7. Fetch API detailed status (depends on prometheus_metrics)
        api_status = await self._get_api_status(prometheus_metrics)

        # 15. Fetch Remediation and Guardian statuses
        remediations = self._get_remediation_statuses(planner)
        guardian_status = self._get_guardian_status(guardian)

        # 16. Determine overall status
        status = "healthy"
        if (
            mlflow_status.stage == "error"
            or mlflow_status.drift_detected
            or api_status.error_rate_5xx > 0.05
            or not rabbitmq_status.connected
            or not redis_status.connected
            or not postgres_status.connected
            or not api_status.reachable
            or not auth_status.reachable
            or not ingestion_status.reachable
            or not portfolio_status.reachable
            or not math_kernel_status.reachable
            or not ml_inference_status.reachable
            or not worker_status.reachable
            or not ray_status.reachable
        ):
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
            auth=auth_status,
            ingestion=ingestion_status,
            portfolio=portfolio_status,
            quant=math_kernel_status,
            inference=ml_inference_status,
            workers=worker_status,
            ray=ray_status,
            remediations=remediations,
            guardian=guardian_status,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    def _get_default_status(self, index: int, error: Exception) -> Any:
        """Returns a default/failed status object for a given index when a fetch fails."""
        logger.error("health_status_fetch_failed", index=index, error=str(error))
        defaults = [
            MLflowStatus(stage="error", drift_detected=False),
            PrometheusMetrics(0.0, 0.0, 0.0, 0.0, 0),
            [],  # redis_anomalies
            RabbitMQStatus(connected=False, queue_depths={}, consumer_counts={}),
            RedisStatus(connected=False, memory_usage_bytes=0, total_keys=0),
            PostgresStatus(connected=False, version="unknown", active_connections=0, hypertables=0, compression_ratio=1.0, job_count=0),
            AuthStatus(reachable=False, p95_latency=0.0, auth_success_rate=0.0, active_tokens=0),
            IngestionStatus(reachable=False, heartbeat_age=9999.0, ticks_per_second=0.0, rejection_rate=0.0),
            PortfolioStatus(reachable=False, positions_count=0, net_delta=0.0, total_vega=0.0, total_gamma=0.0),
            MathKernelStatus(reachable=False, avg_latency_ms=0.0, requests_per_sec=0.0, error_rate=0.0),
            MLInferenceStatus(reachable=False, model_loaded=False, avg_latency_ms=0.0, requests_per_sec=0.0),
            WorkerStatus(reachable=False, broker_connected=False, active_workers=0, queue_backlog={}, avg_task_latency_ms=0.0),
            RayStatus(reachable=False, nodes_alive=0, worker_count=0),
        ]
        return defaults[index]

    def _get_mlflow_status(self) -> MLflowStatus:
        if not self.mlflow_client:
            return MLflowStatus(stage="unknown", drift_detected=False)

        try:
            # Get latest run from default experiment
            runs = self.mlflow_client.search_runs(
                experiment_ids=["0"], max_results=1, order_by=["attribute.start_time DESC"]
            )

            if not runs:
                return MLflowStatus(stage="unknown", drift_detected=False)

            latest_run = runs[0]
            drift_detected = latest_run.data.tags.get("drift_detected", "false").lower() == "true"
            stage = latest_run.data.tags.get("stage", "development")

            return MLflowStatus(
                stage=stage, drift_detected=drift_detected, last_run_id=latest_run.info.run_id
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
            error_rate, latency, cpu_res, mem_res = results

            cpu_usage = float(cpu_res[0]["value"][1]) if cpu_res else 0.0
            mem_usage = float(mem_res[0]["value"][1]) if mem_res else 0.0

            # Fetch request count (Sum of rates)
            req_count_query = (
                f'sum(rate(http_requests_total{{service="{self.api_service_name}"}}[5m]))'
            )
            req_count_res = await asyncio.to_thread(
                self.prometheus_client.prom.custom_query, query=req_count_query
            )
            request_count = 0
            if req_count_res:
                request_count = int(float(req_count_res[0]["value"][1]))

            return PrometheusMetrics(
                error_rate_5xx=error_rate,
                p95_latency=latency,
                cpu_usage=cpu_usage,
                memory_usage=mem_usage,
                request_count=request_count,
            )
        except Exception as e:
            logger.error("prometheus_metrics_fetch_failed", error=str(e))
            return PrometheusMetrics(0.0, 0.0, 0.0, 0.0, 0)

    async def _get_api_status(self, metrics: PrometheusMetrics) -> APIStatus:
        """Determines if the API is reachable and healthy based on metrics."""
        reachable = True

        if metrics.request_count == 0 and metrics.error_rate_5xx == 0:
            reachable = False

        return APIStatus(
            reachable=reachable,
            p95_latency=round(metrics.p95_latency, 4),
            error_rate_5xx=round(metrics.error_rate_5xx, 4),
            request_count=metrics.request_count,
        )

    async def _get_auth_status(self) -> AuthStatus:
        """Fetches detailed Auth service metrics from Prometheus."""
        try:
            # 1. Reachability (based on request rate)
            query_rate = f'sum(rate(http_requests_total{{service="{self.auth_service_name}"}}[5m]))'
            rate_res = await asyncio.to_thread(
                self.prometheus_client.prom.custom_query, query=query_rate
            )
            reachable = len(rate_res) > 0

            # 2. Latency
            latency = await asyncio.to_thread(
                self.prometheus_client.get_p95_latency, self.auth_service_name
            )

            # 3. Success Rate
            # Assuming auth-service exports success/failure counters
            success_query = 'sum(rate(auth_requests_total{status="success"}[5m])) / sum(rate(auth_requests_total[5m]))'
            success_res = await asyncio.to_thread(
                self.prometheus_client.prom.custom_query, query=success_query
            )
            success_rate = float(success_res[0]["value"][1]) if success_res else 1.0

            # 4. Active Tokens (Extracted from Prometheus gauge)
            token_query = "sum(auth_active_tokens_gauge)"
            token_res = await asyncio.to_thread(
                self.prometheus_client.prom.custom_query, query=token_query
            )
            active_tokens = int(float(token_res[0]["value"][1])) if token_res else 0

            return AuthStatus(
                reachable=reachable,
                p95_latency=round(latency, 4),
                auth_success_rate=round(success_rate, 4),
                active_tokens=active_tokens,
            )
        except Exception as e:
            logger.error("auth_status_fetch_failed", error=str(e))
            return AuthStatus(
                reachable=False, p95_latency=0.0, auth_success_rate=0.0, active_tokens=0
            )

    async def _get_ingestion_status(self) -> IngestionStatus:
        """Checks for ingestion service heartbeat and throughput metrics."""

        heartbeat_file = "/tmp/ingestion_heartbeat"
        reachable = False
        heartbeat_age = 9999.0

        try:
            if os.path.exists(heartbeat_file):
                mtime = os.path.getmtime(heartbeat_file)
                heartbeat_age = time.time() - mtime
                reachable = heartbeat_age < 60  # Fresh if within 1 min

            # 1. Ticks per second
            query_tps = (
                f'sum(rate(ingestion_ticks_total{{service="{self.ingestion_service_name}"}}[5m]))'
            )
            tps_res = await asyncio.to_thread(
                self.prometheus_client.prom.custom_query, query=query_tps
            )
            ticks_per_second = float(tps_res[0]["value"][1]) if tps_res else 0.0

            # 2. Rejection Rate
            reject_query = "sum(rate(ingestion_ticks_rejected_total[5m])) / sum(rate(ingestion_ticks_total[5m]))"
            reject_res = await asyncio.to_thread(
                self.prometheus_client.prom.custom_query, query=reject_query
            )
            rejection_rate = float(reject_res[0]["value"][1]) if reject_res else 0.0

            return IngestionStatus(
                reachable=reachable,
                heartbeat_age=round(heartbeat_age, 2),
                ticks_per_second=round(ticks_per_second, 2),
                rejection_rate=round(rejection_rate, 4),
            )
        except Exception as e:
            logger.error("ingestion_status_fetch_failed", error=str(e))
            return IngestionStatus(
                reachable=False, heartbeat_age=9999.0, ticks_per_second=0.0, rejection_rate=0.0
            )

    async def _get_portfolio_status(self) -> PortfolioStatus:
        """Checks for portfolio service reachability and Greek exposure metrics."""
        try:
            # 1. Reachability (HTTP ping or gRPC)
            query_rate = (
                f'sum(rate(http_requests_total{{service="{self.portfolio_service_name}"}}[5m]))'
            )
            rate_res = await asyncio.to_thread(
                self.prometheus_client.prom.custom_query, query=query_rate
            )
            reachable = len(rate_res) > 0

            # 2. Position Count
            pos_query = "sum(portfolio_positions_count)"
            pos_res = await asyncio.to_thread(
                self.prometheus_client.prom.custom_query, query=pos_query
            )
            positions_count = int(float(pos_res[0]["value"][1])) if pos_res else 0

            # 3. Greeks (Delta, Gamma, Vega)
            delta_query = "sum(portfolio_net_delta)"
            vega_query = "sum(portfolio_total_vega)"
            gamma_query = "sum(portfolio_total_gamma)"

            d_res = await asyncio.to_thread(
                self.prometheus_client.prom.custom_query, query=delta_query
            )
            v_res = await asyncio.to_thread(
                self.prometheus_client.prom.custom_query, query=vega_query
            )
            g_res = await asyncio.to_thread(
                self.prometheus_client.prom.custom_query, query=gamma_query
            )

            net_delta = float(d_res[0]["value"][1]) if d_res else 0.0
            total_vega = float(v_res[0]["value"][1]) if v_res else 0.0
            total_gamma = float(g_res[0]["value"][1]) if g_res else 0.0

            return PortfolioStatus(
                reachable=reachable,
                positions_count=positions_count,
                net_delta=round(net_delta, 2),
                total_vega=round(total_vega, 2),
                total_gamma=round(total_gamma, 4),
            )
        except Exception as e:
            logger.error("portfolio_status_fetch_failed", error=str(e))
            return PortfolioStatus(
                reachable=False, positions_count=0, net_delta=0.0, total_vega=0.0, total_gamma=0.0
            )

    async def _get_math_kernel_status(self) -> MathKernelStatus:
        """Checks for pricing engine reachability and performance metrics."""
        try:
            # 1. Reachability (gRPC or HTTP)
            query_rate = (
                f'sum(rate(http_requests_total{{service="{self.math_kernel_service_name}"}}[5m]))'
            )
            rate_res = await asyncio.to_thread(
                self.prometheus_client.prom.custom_query, query=query_rate
            )
            reachable = len(rate_res) > 0

            # 2. Avg Pricing Latency (ms)
            latency_query = "avg(pricing_computation_time_ms)"
            lat_res = await asyncio.to_thread(
                self.prometheus_client.prom.custom_query, query=latency_query
            )
            avg_latency = float(lat_res[0]["value"][1]) if lat_res else 0.0

            # 3. Requests per second
            tps_query = "sum(rate(pricing_requests_total[5m]))"
            tps_res = await asyncio.to_thread(
                self.prometheus_client.prom.custom_query, query=tps_query
            )
            requests_per_sec = float(tps_res[0]["value"][1]) if tps_res else 0.0

            # 4. Error Rate
            err_query = (
                "sum(rate(pricing_errors_total[5m])) / sum(rate(pricing_requests_total[5m]))"
            )
            err_res = await asyncio.to_thread(
                self.prometheus_client.prom.custom_query, query=err_query
            )
            error_rate = (
                float(err_res[0]["value"][1])
                if err_res and err_res[0]["value"][1] != "NaN"
                else 0.0
            )

            return MathKernelStatus(
                reachable=reachable,
                avg_latency_ms=round(avg_latency, 2),
                requests_per_sec=round(requests_per_sec, 2),
                error_rate=round(error_rate, 4),
            )
        except Exception as e:
            logger.error("math_kernel_status_fetch_failed", error=str(e))
            return MathKernelStatus(
                reachable=False, avg_latency_ms=0.0, requests_per_sec=0.0, error_rate=0.0
            )

    async def _get_ml_inference_status(self) -> MLInferenceStatus:
        """Checks for ML inference service reachability and model performance."""
        try:
            # 1. Reachability
            query_rate = (
                f'sum(rate(http_requests_total{{service="{self.ml_inference_service_name}"}}[5m]))'
            )
            rate_res = await asyncio.to_thread(
                self.prometheus_client.prom.custom_query, query=query_rate
            )
            reachable = len(rate_res) > 0

            # 2. Inference Latency (ms)
            latency_query = "avg(onnx_inference_latency_ms)"
            lat_res = await asyncio.to_thread(
                self.prometheus_client.prom.custom_query, query=latency_query
            )
            avg_latency = float(lat_res[0]["value"][1]) if lat_res else 0.0

            # 3. Model Loaded Check
            # This is available via Prometheus gauge if we add it, otherwise assume reachable implies loaded
            model_loaded = reachable

            # 4. Throughput (req/s)
            tps_query = f'sum(rate(http_requests_total{{service="{self.ml_inference_service_name}",path="/predict"}}[5m]))'
            tps_res = await asyncio.to_thread(
                self.prometheus_client.prom.custom_query, query=tps_query
            )
            requests_per_sec = float(tps_res[0]["value"][1]) if tps_res else 0.0

            return MLInferenceStatus(
                reachable=reachable,
                model_loaded=model_loaded,
                avg_latency_ms=round(avg_latency, 2),
                requests_per_sec=round(requests_per_sec, 2),
            )
        except Exception as e:
            logger.error("ml_inference_status_fetch_failed", error=str(e))
            return MLInferenceStatus(
                reachable=False, model_loaded=False, avg_latency_ms=0.0, requests_per_sec=0.0
            )

    async def _get_worker_status(self) -> WorkerStatus:
        """Checks Celery worker health and queue status."""
        try:
            from src.workers.tasks.celery_app import celery_app
            if celery_app is None:
                raise ImportError("Celery app not initialized")
        except (ImportError, Exception):
            logger.warning("celery_not_available_skipping_status")
            return WorkerStatus(
                reachable=False,
                broker_connected=False,
                active_workers=0,
                queue_backlog={},
                avg_task_latency_ms=0.0,
            )

        try:
            # 1. Reachability & Broker
            reachable = False
            with celery_app.connection() as conn:
                conn.ensure_connection(max_retries=1)
                reachable = True

            # 2. Active Workers (from Celery Inspector)
            # This is slow, so we use a short timeout or rely on Prometheus in prod
            inspector = celery_app.control.inspect(timeout=1.0)
            pings = inspector.ping()
            active_workers = len(pings) if pings else 0

            # 3. Queue Backlog (from Prometheus if available, else RabbitMQ report)
            q_backlog = {}
            # We already fetch queue depths in _get_rabbitmq_status, but here we group them for workers
            rmq_report = await self._get_rabbitmq_status()
            q_backlog = rmq_report.queue_depths

            # 4. Task Latency (from Prometheus)
            lat_query = "avg(celery_task_wait_time_seconds)"
            lat_res = await asyncio.to_thread(
                self.prometheus_client.prom.custom_query, query=lat_query
            )
            avg_latency = float(lat_res[0]["value"][1]) * 1000 if lat_res else 0.0

            return WorkerStatus(
                reachable=reachable,
                broker_connected=reachable,
                active_workers=active_workers,
                queue_backlog=q_backlog,
                avg_task_latency_ms=round(avg_latency, 2),
            )
        except Exception as e:
            logger.error("worker_status_fetch_failed", error=str(e))
            return WorkerStatus(
                reachable=False,
                broker_connected=False,
                active_workers=0,
                queue_backlog={},
                avg_task_latency_ms=0.0,
            )

    async def _get_ray_status(self) -> RayStatus:
        """Checks Ray cluster health and actor availability."""
        if not RAY_AVAILABLE:
            return RayStatus(reachable=False, nodes_alive=0, worker_count=0)
        try:
            if not ray.is_initialized():
                # Don't initialize here to avoid side effects if head node is down
                reachable = False
                alive_nodes = 0
            else:
                reachable = True
                nodes = ray.nodes()
                alive_nodes = len([n for n in nodes if n["Alive"]])

            # Metrics from Prometheus (assuming ray-exporter is used)
            # count(ray_actor_status{status="ALIVE"})
            count_query = "sum(ray_actor_count)"
            count_res = await asyncio.to_thread(
                self.prometheus_client.prom.custom_query, query=count_query
            )
            worker_count = int(float(count_res[0]["value"][1])) if count_res else 0

            return RayStatus(
                reachable=reachable, nodes_alive=alive_nodes, worker_count=worker_count
            )
        except Exception as e:
            logger.error("ray_status_fetch_failed", error=str(e))
            return RayStatus(reachable=False, nodes_alive=0, worker_count=0)

    async def _get_redis_anomalies(self) -> list[RedisAnomaly]:
        try:
            redis = await get_redis_client()
            # Get last 10 anomalies from history
            raw_anomalies = await redis.lrange(self.anomaly_history_key, 0, 9)

            anomalies = []
            for raw in raw_anomalies:
                try:
                    data = msgspec.json.decode(raw)
                    anomalies.append(
                        RedisAnomaly(
                            timestamp=data.get("timestamp", ""),
                            description=data.get("description", ""),
                            severity=data.get("severity", "medium"),
                        )
                    )
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
                connected=connected, queue_depths=queue_depths, consumer_counts=consumer_counts
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
                connected=True, memory_usage_bytes=memory_usage, total_keys=key_count
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
                jobs = await conn.execute(text("SELECT count(*) FROM timescaledb_information.jobs"))
                job_count = jobs.scalar() or 0

                # 4. Compression ratio (Safer cross-version check)
                try:
                    compression = await conn.execute(
                        text("""
                        SELECT 
                            COALESCE(SUM(uncompressed_total_bytes) / NULLIF(SUM(compressed_total_bytes), 0), 1.0)
                        FROM timescaledb_information.compression_settings
                    """)
                    )
                    ratio = float(compression.scalar() or 1.0)
                except Exception:
                    ratio = 1.0

                return PostgresStatus(
                    connected=True,
                    version=version.split(",")[0],  # Short version string
                    active_connections=active_connections,
                    hypertables=int(hyper_count),
                    compression_ratio=round(ratio, 2),
                    job_count=int(job_count),
                )
        except Exception as e:
            logger.error("postgres_status_fetch_failed", error=str(e))
            return PostgresStatus(
                connected=False,
                version="unknown",
                active_connections=0,
                hypertables=0,
                compression_ratio=1.0,
                job_count=0,
            )

    def _get_remediation_statuses(self, planner: Any | None) -> list[RemediationStatus]:
        """Collects status info from all registered remediators."""
        if not planner:
            return []

        statuses = []
        for r in planner.remediators.values():
            statuses.append(
                RemediationStatus(
                    name=r.name,
                    status=r.get_status(),
                    last_run=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(r.last_run))
                    if r.last_run > 0
                    else "never",
                )
            )
        return statuses

    def _get_guardian_status(self, guardian: Any | None) -> GuardianStatus:
        """Collects status from the Autonomous Guardian."""
        if not guardian:
            return GuardianStatus(active=False, safe_mode=False, paused_features=[])

        return GuardianStatus(
            active=guardian.is_active,
            safe_mode=getattr(guardian, "is_safe_mode", False),
            paused_features=getattr(guardian, "paused_features", []),
        )