import asyncio
import time
from typing import Any

import pandas as pd
import structlog

from src.ml.aiops.anomaly_detector import AnomalyDetector
from src.ml.aiops.prometheus_adapter import PrometheusClient
from src.ml.aiops.remediators import BaseRemediator, RemediationPlanner
from src.ml.aiops.health_reporter import HealthReporter
from src.ml.drift import calculate_ks_test, calculate_psi
from src.ml.forecasting.tft_model import PriceTFTModel
from src.shared.observability import (
    post_grafana_annotation,
    setup_logging,
)

from src.ml.aiops.autonomous_guardian import AutonomousGuardian

logger = structlog.get_logger(__name__)

class AutonomousEngine:
    """
    Autonomous orchestrator that combines real-time anomaly detection
    with distribution-based drift analysis, automated remediation,
    and high-level guardian oversight.
    """

    def __init__(
        self,
        detector: AnomalyDetector | None = None,
        remediators: list[BaseRemediator] | None = None,
        config: dict[str, Any] | None = None,
        check_interval: int = 10,
        drift_threshold_psi: float = 0.2,
    ):
        setup_logging()
        self.config = config or {}
        
        # 1. Detection Core
        self.detector = detector or AnomalyDetector(
            engine="transformer", 
            input_dim=self.config.get("transformer_input_dim", 10)
        )
        self.remediators = remediators or RemediationPlanner().remediators.values()
        self.planner = RemediationPlanner(list(self.remediators))
        
        # 2. Thresholds and State
        self.check_interval = self.config.get("check_interval_seconds", check_interval)
        self.drift_threshold_psi = self.config.get("data_drift_psi_threshold", drift_threshold_psi)
        self.is_running = False
        self.reference_data = None
        self.history = []
        self.anomaly_queue_key = "aiops:anomaly_queue"

        # 3. Enhanced Detectors
        self.prometheus_url = self.config.get("prometheus_url")
        self.prometheus_client = (
            PrometheusClient(url=self.prometheus_url) if self.prometheus_url else None
        )
        self.api_service_name = self.config.get("api_service_name", "bsopt-api")

        ae_input_dim = self.config.get("autoencoder_input_dim")
        self.autoencoder_detector = (
            AnomalyDetector(engine="autoencoder", input_dim=ae_input_dim) if ae_input_dim else None
        )

        self.forecaster = (
            PriceTFTModel(config=self.config.get("tft_config"))
            if self.config.get("predictive_scaling_enabled")
            else None
        )
        self.max_history = 1000
        self.last_baseline_update = time.time()
        
        # 4. Guardian Oversight
        self.guardian = AutonomousGuardian(self)

        # 5. Health Reporting
        self.health_reporter = HealthReporter(
            prometheus_url=self.prometheus_url,
            api_service_name=self.api_service_name
        ) if self.prometheus_url else None

    async def _process_redis_anomalies(self):
        """Polls Redis for externally reported anomalies (e.g. from Webhooks)."""
        from src.shared.utils.cache import get_redis
        redis = get_redis()
        if not redis:
            return []

        anomalies = []
        while True:
            data = await redis.lpop(self.anomaly_queue_key)
            if not data:
                break
            try:
                import msgspec
                anomaly = msgspec.json.decode(data)
                anomalies.append(anomaly)
                logger.warning("external_anomaly_received", anomaly=anomaly)
            except Exception as e:
                logger.error("failed_to_decode_external_anomaly", error=str(e))
        return anomalies

    async def run_cycle(self, current_data: pd.DataFrame | None = None):
        """
        Perform one cycle of detection, drift analysis, and remediation.
        """
        logger.info("self_healing_cycle_start")

        try:
            # 1. Gather all anomaly signals
            tasks = [self._process_redis_anomalies()]
            
            if current_data is not None:
                tasks.append(asyncio.to_thread(self.detector.detect, current_data))
                tasks.append(asyncio.to_thread(self._analyze_drift, current_data))
            
            if self.prometheus_client:
                tasks.append(self._detect_system_anomalies())
            
            if self.health_reporter:
                tasks.append(self.health_reporter.get_health_report(
                    planner=self.planner,
                    guardian=self.guardian
                ))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Extract results safely
            external_anomalies = results[0] if not isinstance(results[0], Exception) else []
            
            idx = 1
            ml_anomalies = []
            if current_data is not None:
                ml_anomalies = results[idx] if not isinstance(results[idx], Exception) else []
                idx += 1
                drift_anomalies = results[idx] if not isinstance(results[idx], Exception) else []
                idx += 1
            else:
                drift_anomalies = []

            system_anomalies = []
            if self.prometheus_client:
                system_anomalies = results[idx] if not isinstance(results[idx], Exception) else []
                idx += 1
            
            if self.health_reporter:
                health_report = results[idx] if not isinstance(results[idx], Exception) else None
                if health_report:
                    logger.info("health_report_status", status=health_report.status)
                    # Publish health report to RabbitMQ (Both Audit and Telemetry)
                    try:
                        import msgspec
                        payload = {
                            "type": "health_report",
                            "status": health_report.status,
                            "data": msgspec.json.decode(msgspec.json.encode(health_report)),
                            "timestamp": health_report.timestamp
                        }
                        # 1. Audit Exchange (for persistence/logging)
                        await self.health_reporter.rmq.publish_audit(payload)
                        # 2. Telemetry Exchange (for real-time monitoring)
                        await self.health_reporter.rmq.publish_telemetry(payload, routing_key=f"telemetry.health.{health_report.status}")
                    except Exception as e:
                        logger.error("health_report_publish_failed", error=str(e))

            all_anomalies = external_anomalies + ml_anomalies + drift_anomalies + system_anomalies

            if not all_anomalies:
                logger.info("system_health_nominal")
                return

            # 2. Remediation Execution
            for anomaly in all_anomalies:
                actions = self.planner.plan(anomaly)
                for action in actions:
                    if action.can_run():
                        logger.warning("executing_remediation", action=action.name, type=anomaly.get("type"))
                        success = await action.remediate(anomaly)
                        await action.update_last_run()
                        self._record_history(action.name, anomaly, success)
                        
                        if success:
                            post_grafana_annotation(
                                f"Remediated {anomaly.get('type')} via {action.name}",
                                ["aiops", "remediation"]
                            )

        except Exception as e:
            logger.error("self_healing_cycle_error", error=str(e))

    def _record_history(self, action: str, anomaly: dict, success: bool):
        if not hasattr(self, "max_history"):
            self.max_history = 1000
        self.history.append(
            {
                "timestamp": time.time(),
                "action": action,
                "anomaly": anomaly.get("type"),
                "success": success,
            }
        )
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def _analyze_drift(self, current_data: pd.DataFrame) -> list[dict]:
        """
        Detects shifts in system metric distributions (e.g. baseline latency shift).
        """
        if self.reference_data is None:
            self.reference_data = current_data
            if not hasattr(self, "last_baseline_update"):
                self.last_baseline_update = time.time()
            logger.info("drift_baseline_initialized")
            return []

        drift_anomalies = []
        numeric_cols = current_data.select_dtypes(include=["number"]).columns

        for col in numeric_cols:
            ref_vals = self.reference_data[col].values
            curr_vals = current_data[col].values

            try:
                psi_score = calculate_psi(ref_vals, curr_vals)
                ks_stat, p_val = calculate_ks_test(ref_vals, curr_vals)

                if psi_score > self.drift_threshold_psi:
                    drift_info = {
                        "type": "distribution_drift",
                        "metric": col,
                        "psi_score": float(psi_score),
                        "ks_p_val": float(p_val),
                        "score": float(psi_score),
                    }
                    drift_anomalies.append(drift_info)
                    logger.warning("metric_distribution_drift_detected", **drift_info)
            except Exception:
                pass

        # Periodically update reference data (every 4 hours)
        now = time.time()
        if not hasattr(self, "last_baseline_update"):
            self.last_baseline_update = now
        if now - self.last_baseline_update > 14400:
            self.reference_data = current_data
            self.last_baseline_update = now
            logger.info("drift_baseline_updated", timestamp=now)
        return drift_anomalies

    async def _detect_system_anomalies(self) -> list[dict]:
        """Scans Prometheus for system-level anomalies."""
        if not self.prometheus_client:
            return []

        system_anomalies = []
        try:
            # 1. Error Rate
            error_rate = await self.prometheus_client.get_5xx_error_rate(self.api_service_name)
            if error_rate > self.config.get("error_rate_threshold", 0.05):
                system_anomalies.append(
                    {"type": "high_error_rate", "metric": error_rate, "severity": "high"}
                )

            # 2. Latency
            p95_latency = await self.prometheus_client.get_p95_latency(self.api_service_name)
            if p95_latency > self.config.get("latency_threshold", 0.5):
                system_anomalies.append(
                    {"type": "high_latency", "metric": p95_latency, "severity": "medium"}
                )

            # 3. Predictive (Forecasting)
            if self.forecaster:
                recent_df = await self.prometheus_client.get_metric_range(
                    self.api_service_name, "container_cpu_usage_seconds_total"
                )
                if not recent_df.empty:
                    forecast = self.forecaster.predict(recent_df)
                    if forecast is not None and forecast.max() > 0.8:
                        system_anomalies.append(
                            {"type": "predicted_load_spike", "forecast_max": float(forecast.max())}
                        )

            # 4. Critical Database Pool Check
            if self.health_reporter:
                report = await self.health_reporter.get_health_report(self.planner, self.guardian)
                
                if report.postgres.active_connections > 50:
                    system_anomalies.append({
                        "type": "db_pool_exhaustion",
                        "metrics": {"total_connections": report.postgres.active_connections},
                        "score": 0.8,
                        "timestamp": time.time(),
                    })
                
                # 5. API-Specific Anomalies (from Health Report)
                if not report.api.reachable:
                    system_anomalies.append({
                        "type": "api_unreachable",
                        "severity": "critical",
                        "timestamp": time.time()
                    })
                if report.api.error_rate_5xx > 0.15:
                    system_anomalies.append({
                        "type": "api_error_spike",
                        "metric": report.api.error_rate_5xx,
                        "severity": "high",
                        "timestamp": time.time()
                    })

                # 6. Auth-Specific Anomalies (from Health Report)
                if not report.auth.reachable:
                    system_anomalies.append({
                        "type": "auth_unreachable",
                        "severity": "critical",
                        "timestamp": time.time()
                    })
                if report.auth.auth_success_rate < 0.9:
                    system_anomalies.append({
                        "type": "auth_failure_spike",
                        "metric": report.auth.auth_success_rate,
                        "severity": "high",
                        "timestamp": time.time()
                    })

                # 7. Ingestion-Specific Anomalies
                if not report.ingestion.reachable or report.ingestion.heartbeat_age > 120:
                    system_anomalies.append({
                        "type": "ingestion_stall",
                        "heartbeat_age": report.ingestion.heartbeat_age,
                        "severity": "critical",
                        "timestamp": time.time()
                    })
                if report.ingestion.rejection_rate > 0.2:
                    system_anomalies.append({
                        "type": "high_data_rejection",
                        "metric": report.ingestion.rejection_rate,
                        "severity": "high",
                        "timestamp": time.time()
                    })

                # 8. Portfolio-Specific Anomalies (Exposure Check)
                if abs(report.portfolio.net_delta) > 1000: # Example threshold
                    system_anomalies.append({
                        "type": "high_risk_exposure",
                        "metric": "net_delta",
                        "value": report.portfolio.net_delta,
                        "severity": "high",
                        "timestamp": time.time()
                    })
                if not report.portfolio.reachable:
                    system_anomalies.append({
                        "type": "portfolio_unreachable",
                        "severity": "critical",
                        "timestamp": time.time()
                    })

                # 9. Math Kernel Latency Anomaly
                if report.quant.avg_latency_ms > 500:
                    system_anomalies.append({
                        "type": "high_pricing_latency",
                        "metric": report.quant.avg_latency_ms,
                        "severity": "medium",
                        "timestamp": time.time()
                    })
                if report.quant.error_rate > 0.05:
                    system_anomalies.append({
                        "type": "kernel_error_spike",
                        "metric": report.quant.error_rate,
                        "severity": "high",
                        "timestamp": time.time()
                    })

        except Exception as e:
            logger.error("system_anomaly_detection_failed", error=str(e))

        return system_anomalies

    async def _ensure_infrastructure_ready(self, timeout: int = 60, interval: int = 5):
        """Blocks until RabbitMQ, Redis, and Postgres/TimescaleDB are reachable."""
        from src.shared.utils.cache import get_redis_client
        from src.database import get_async_engine
        from sqlalchemy import text
        
        logger.info("waiting_for_infrastructure_readiness", timeout=timeout)
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # 1. Check RabbitMQ
                if self.health_reporter and self.health_reporter.rmq:
                    await self.health_reporter.rmq.connect()
                
                # 2. Check Redis
                redis = await get_redis_client()
                await redis.ping()
                
                # 3. Check TimescaleDB
                engine = get_async_engine()
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))

                # 4. Check API (Internal REST Gateway)
                await self._check_api_ready()

                # 5. Check Auth (Internal Security Gateway)
                await self._check_auth_ready()

                # 6. Check Ingestion (Data Pipeline Gateway)
                await self._check_ingestion_ready()

                # 7. Check Portfolio (Risk & Exposure Gateway)
                await self._check_portfolio_ready()

                # 8. Check Math Kernel (Pricing & Computation Gateway)
                await self._check_math_kernel_ready()

                logger.info("infrastructure_ready")
                return
            except Exception:
                logger.warning("infrastructure_not_ready_retrying", 
                               elapsed=int(time.time() - start_time))
                await asyncio.sleep(interval)
        
        logger.error("infrastructure_readiness_timeout_proceeding_degraded")

    async def _check_api_ready(self) -> bool:
        """Polls the API health endpoint until it's ready."""
        import httpx
        # We assume the service name is resolvable in the internal network
        url = f"http://{self.api_service_name}:8000/health"
        logger.info("polling_api_readiness", url=url)
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, timeout=2.0)
                if resp.status_code == 200:
                    logger.info("api_ready")
                    return True
            except Exception as e:
                logger.debug("api_ping_failed", error=str(e))
        
        raise RuntimeError(f"API at {url} is not yet reachable")

    async def _check_auth_ready(self) -> bool:
        """Polls the Auth health endpoint until it's ready."""
        import httpx
        url = "http://auth:3001/health" # Primary auth gateway
        logger.info("polling_auth_readiness", url=url)
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, timeout=2.0)
                if resp.status_code == 200:
                    logger.info("auth_ready")
                    return True
            except Exception as e:
                logger.debug("auth_ping_failed", error=str(e))
        
        raise RuntimeError(f"Auth Service at {url} is not yet reachable")

    async def _check_ingestion_ready(self) -> bool:
        """Polls the Ingestion heartbeat until it's ready."""
        import os
        heartbeat_file = "/tmp/ingestion_heartbeat"
        logger.info("polling_ingestion_readiness", file=heartbeat_file)
        
        if os.path.exists(heartbeat_file):
            age = time.time() - os.path.getmtime(heartbeat_file)
            if age < 60:
                logger.info("ingestion_ready")
                return True
        
        # If file missing or too old, check gRPC port as fallback
        import socket
        try:
            with socket.create_connection(("localhost", 50053), timeout=2.0):
                logger.info("ingestion_grpc_ready")
                return True
        except Exception:
            pass
            
        raise RuntimeError(f"Ingestion Service is not reporting heartbeats at {heartbeat_file}")

    async def _check_portfolio_ready(self) -> bool:
        """Polls the Portfolio health endpoint until it's ready."""
        import httpx
        url = "http://portfolio:8080/health" # Port 8080 for portfolio internal gateway
        logger.info("polling_portfolio_readiness", url=url)
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, timeout=2.0)
                if resp.status_code == 200:
                    logger.info("portfolio_ready")
                    return True
            except Exception as e:
                logger.debug("portfolio_ping_failed", error=str(e))
        
        raise RuntimeError(f"Portfolio Service at {url} is not yet reachable")

    async def _check_math_kernel_ready(self) -> bool:
        """Polls the Math Kernel health endpoint until it's ready."""
        import httpx
        url = "http://math-kernel:8080/health" # Port 8080 for pricing internal gateway
        logger.info("polling_math_kernel_readiness", url=url)
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, timeout=2.0)
                if resp.status_code == 200:
                    logger.info("math_kernel_ready")
                    return True
            except Exception as e:
                logger.debug("math_kernel_ping_failed", error=str(e))
        
        raise RuntimeError(f"Math Kernel at {url} is not yet reachable")

    async def start(self, data_source: Any):
        """Start the autonomous self-healing loop and guardian oversight."""
        self.is_running = True
        logger.info("autonomous_engine_started")

        # Wait for RabbitMQ/Infrastructure
        await self._ensure_infrastructure_ready()

        # Start Guardian in a separate task
        asyncio.create_task(self.guardian.monitor_integrity())

        while self.is_running:
            try:
                if hasattr(data_source, "get_latest_metrics_async"):
                    data = await data_source.get_latest_metrics_async()
                elif asyncio.iscoroutinefunction(data_source.get_latest_metrics):
                    data = await data_source.get_latest_metrics()
                else:
                    data = await asyncio.to_thread(data_source.get_latest_metrics)

                if isinstance(data, pd.DataFrame):
                    await self.run_cycle(data)
                else:
                    logger.error("invalid_data_format", type=type(data))
            except Exception as e:
                logger.error("loop_iteration_error", error=str(e))

            await asyncio.sleep(self.check_interval)

    def stop(self):
        """Stop the orchestrator loop and cleanup resources."""
        self.is_running = False
        if hasattr(self.detector, "shutdown"):
            self.detector.shutdown()
        if self.forecaster and hasattr(self.forecaster, "close"):
            self.forecaster.close()
        logger.info("self_healing_orchestrator_stopped")
