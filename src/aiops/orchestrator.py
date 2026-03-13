import asyncio

import structlog

from src.aiops.remediators import RemediationPlanner
from src.utils.cache import get_redis

logger = structlog.get_logger(__name__)

class AutonomousHealthOrchestrator:
    """
    Orchestrates automated system recovery and healing.
    Monitors anomaly signals (from OTel/Prometheus via Redis/Webhooks)
    and executes remediation plans.
    """
    
    def __init__(self):
        self.planner = RemediationPlanner()
        self.running = False
        self.anomaly_queue_key = "aiops:anomaly_queue"
        
    async def start(self):
        """Starts the autonomous healing loop."""
        self.running = True
        logger.info("autonomous_health_orchestrator_started")
        
        while self.running:
            try:
                await self._process_anomalies()
                await asyncio.sleep(10) # Poll every 10s
            except Exception as e:
                logger.error("orchestrator_loop_failed", error=str(e))
                await asyncio.sleep(30) # Backoff on error
                
    async def _process_anomalies(self):
        """Polls for new anomalies and triggers remediations."""
        redis = get_redis()
        if not redis:
            return
            
        # Get pending anomalies (e.g., from Prometheus Alertmanager webhooks)
        anomaly_data = await redis.lpop(self.anomaly_queue_key)
        if not anomaly_data:
            return
            
        import msgspec
        anomaly = msgspec.json.decode(anomaly_data)
        logger.warning("anomaly_detected_triggering_remidiation", anomaly=anomaly)
        
        # 1. Plan Remediations
        actions = self.planner.plan(anomaly)
        
        # 2. Execute Remediations
        for action in actions:
            logger.info("executing_remediation_action", action=action.name)
            success = await action.remediate(anomaly)
            
            if success:
                # 3. Optional Validation
                is_valid = await action.validate(anomaly)
                if is_valid:
                    logger.info("remediation_successful_and_validated", action=action.name)
                    await action.update_last_run()
                else:
                    logger.error("remediation_validation_failed", action=action.name)
            else:
                logger.error("action_execution_failed", action=action.name)

    def stop(self):
        self.running = False
        logger.info("autonomous_health_orchestrator_stopped")
