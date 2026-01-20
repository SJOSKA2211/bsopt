import unittest
from unittest.mock import MagicMock, patch, call
import time
from src.aiops.aiops_orchestrator import AIOpsOrchestrator

class TestAIOpsOrchestratorRemediation(unittest.TestCase):
    def setUp(self):
        self.default_config = {
            "check_interval_seconds": 60,
            "prometheus_url": "http://localhost:9090",
            "api_service_name": "test_api",
            "redis_cache_pattern": "test_pattern:*"
        }
        
        # Patch external dependencies
        self.patcher_prometheus = patch('src.aiops.aiops_orchestrator.PrometheusClient')
        self.patcher_docker = patch('src.aiops.aiops_orchestrator.DockerRemediator')
        self.patcher_redis = patch('src.aiops.aiops_orchestrator.RedisRemediator')
        self.patcher_ml_trigger = patch('src.aiops.aiops_orchestrator.MLPipelineTrigger')
        self.patcher_post_annotation = patch('src.aiops.aiops_orchestrator.post_grafana_annotation')
        self.patcher_logger = patch('src.aiops.aiops_orchestrator.logger')
        self.patcher_setup_logging = patch('src.aiops.aiops_orchestrator.setup_logging')
        self.patcher_push_metrics = patch('src.aiops.aiops_orchestrator.push_metrics')

        self.mock_prometheus = self.patcher_prometheus.start()
        self.mock_docker = self.patcher_docker.start()
        self.mock_redis = self.patcher_redis.start()
        self.mock_ml_trigger = self.patcher_ml_trigger.start()
        self.mock_post_annotation = self.patcher_post_annotation.start()
        self.mock_logger = self.patcher_logger.start()
        self.mock_setup_logging = self.patcher_setup_logging.start()
        self.mock_push_metrics = self.patcher_push_metrics.start()

    def tearDown(self):
        patch.stopall()

    def test_remediate_high_error_rate(self):
        orchestrator = AIOpsOrchestrator(self.default_config)
        anomalies = {"high_error_rate": True}
        
        orchestrator._remediate_anomalies(anomalies)
        
        self.mock_docker.return_value.restart_service.assert_called_with("test_api")
        self.mock_post_annotation.assert_called_with(
            "Remediation: Restarting 'test_api' due to high error rate.",
            ["remediation", "api_spike", "test_api"]
        )

    def test_remediate_high_latency(self):
        orchestrator = AIOpsOrchestrator(self.default_config)
        anomalies = {"high_latency": True}
        
        orchestrator._remediate_anomalies(anomalies)
        
        self.mock_docker.return_value.restart_service.assert_called_with("test_api")
        self.mock_post_annotation.assert_called_with(
            "Remediation: Restarting 'test_api' due to high latency.",
            ["remediation", "api_spike", "test_api"]
        )

    def test_remediate_data_drift(self):
        orchestrator = AIOpsOrchestrator(self.default_config)
        anomalies = {"data_drift": True}
        
        orchestrator._remediate_anomalies(anomalies)
        
        self.mock_ml_trigger.return_value.trigger_retraining.assert_called_once()
        self.mock_post_annotation.assert_called_with(
            "Remediation: Triggering ML pipeline retraining due to data drift.",
            ["remediation", "data_drift"]
        )

    def test_remediate_univariate_anomaly(self):
        orchestrator = AIOpsOrchestrator(self.default_config)
        anomalies = {"univariate_anomaly": True}
        
        orchestrator._remediate_anomalies(anomalies)
        
        self.mock_redis.return_value.purge_cache.assert_called_with("test_pattern:*")
        self.mock_post_annotation.assert_called_with(
            "Remediation: Purging Redis cache due to univariate anomaly.",
            ["remediation", "anomaly", "redis_cache"]
        )

    def test_remediate_multivariate_anomaly(self):
        orchestrator = AIOpsOrchestrator(self.default_config)
        anomalies = {"multivariate_anomaly": True}
        
        orchestrator._remediate_anomalies(anomalies)
        
        self.mock_redis.return_value.purge_cache.assert_called_with("test_pattern:*")
        self.mock_post_annotation.assert_called_with(
            "Remediation: Purging Redis cache due to multivariate anomaly.",
            ["remediation", "anomaly", "redis_cache"]
        )

    @patch('time.sleep')
    def test_run_loop_with_anomalies(self, mock_sleep):
        orchestrator = AIOpsOrchestrator(self.default_config)
        
        # Mock detect anomalies to return anomalies then nothing
        orchestrator._detect_anomalies = MagicMock(side_effect=[
            {"high_error_rate": True}, 
            {}
        ])
        
        orchestrator.run(iterations=2)
        
        self.assertEqual(orchestrator._detect_anomalies.call_count, 2)
        self.mock_docker.return_value.restart_service.assert_called_once()
        self.mock_push_metrics.assert_called()
        self.assertEqual(mock_sleep.call_count, 1) # Should sleep once between iterations

    @patch('time.sleep')
    def test_run_loop_exception_handling(self, mock_sleep):
        orchestrator = AIOpsOrchestrator(self.default_config)
        
        # Mock detect anomalies to raise exception
        orchestrator._detect_anomalies = MagicMock(side_effect=Exception("Test Error"))
        
        orchestrator.run(iterations=1)
        
        self.mock_logger.error.assert_called_with("aiops_orchestrator_loop_error", error="Test Error")
        self.mock_push_metrics.assert_called()

if __name__ == '__main__':
    unittest.main()
