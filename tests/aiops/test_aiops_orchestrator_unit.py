import tests.mock_all
import pytest
from unittest.mock import MagicMock, patch, ANY
from src.aiops.aiops_orchestrator import AIOpsOrchestrator

@pytest.fixture
def mock_config():
    return {
        "prometheus_url": "http://prometheus:9090",
        "api_service_name": "bsopt-api",
        "check_interval_seconds": 0.1,
        "autoencoder_input_dim": 10,
        "ml_pipeline_config": {
            "ticker": "AAPL",
            "framework": "xgboost"
        }
    }

def test_orchestrator_init(mock_config):
    with patch("src.aiops.aiops_orchestrator.PrometheusClient"):
        with patch("src.aiops.aiops_orchestrator.IsolationForestDetector"):
            with patch("src.aiops.aiops_orchestrator.PriceTFTModel"):
                with patch("src.aiops.aiops_orchestrator.AutoencoderDetector"):
                    with patch("src.aiops.aiops_orchestrator.TransformerAnomalyDetector"):
                        orchestrator = AIOpsOrchestrator(mock_config)
                        assert orchestrator.prometheus_url == "http://prometheus:9090"
                        assert orchestrator.autoencoder_detector is not None

def test_detect_anomalies_high_error_rate(mock_config):
    with patch("src.aiops.aiops_orchestrator.PrometheusClient") as MockProm:
        with patch("src.aiops.aiops_orchestrator.IsolationForestDetector"):
            with patch("src.aiops.aiops_orchestrator.AutoencoderDetector"):
                with patch("src.aiops.aiops_orchestrator.DataDriftDetector"):
                    with patch("src.aiops.aiops_orchestrator.TransformerAnomalyDetector"):
                        mock_prom_instance = MockProm.return_value
                        # High error rate
                        mock_prom_instance.get_5xx_error_rate.return_value = 0.1 # > 0.05
                        mock_prom_instance.get_p95_latency.return_value = 0.1 # < 0.5
                        mock_prom_instance.get_historical_metric_data.return_value = None
                        
                        orchestrator = AIOpsOrchestrator(mock_config)
                        anomalies = orchestrator._detect_anomalies()
                        
                        assert "high_error_rate" in anomalies
                        assert "high_latency" not in anomalies

def test_detect_anomalies_ml(mock_config):
    with patch("src.aiops.aiops_orchestrator.PrometheusClient") as MockProm:
        with patch("src.aiops.aiops_orchestrator.IsolationForestDetector") as MockIF:
            with patch("src.aiops.aiops_orchestrator.AutoencoderDetector") as MockAE:
                with patch("src.aiops.aiops_orchestrator.DataDriftDetector") as MockDrift:
                    with patch("src.aiops.aiops_orchestrator.TransformerAnomalyDetector") as MockTrans:
                        mock_prom_instance = MockProm.return_value
                        mock_prom_instance.get_5xx_error_rate.return_value = 0.0
                        mock_prom_instance.get_p95_latency.return_value = 0.0
                        mock_prom_instance.get_historical_metric_data.return_value = [1, 2, 3]
                        mock_prom_instance.get_historical_metric_data_multi.return_value = [[1], [2]]
                        
                        # Isolation Forest anomaly
                        MockIF.return_value.fit_predict.return_value = [-1]
                        # AE anomaly
                        MockAE.return_value.fit_predict.return_value = [-1]
                        # Transformer anomaly
                        MockTrans.return_value.detect.return_value = {"is_anomaly": True, "score": 0.9}
                        # Drift anomaly
                        MockDrift.return_value.detect_drift.return_value = (True, {"p_value": 0.01})
                        
                        orchestrator = AIOpsOrchestrator(mock_config)
                        anomalies = orchestrator._detect_anomalies()
                        
                        assert "univariate_anomaly" in anomalies
                        assert "multivariate_anomaly" in anomalies
                        assert "transformer_anomaly" in anomalies
                        assert "data_drift" in anomalies

def test_remediate_anomalies(mock_config):
    with patch("src.aiops.aiops_orchestrator.PrometheusClient"):
        with patch("src.aiops.aiops_orchestrator.IsolationForestDetector"):
            with patch("src.aiops.aiops_orchestrator.AutoencoderDetector"):
                with patch("src.aiops.aiops_orchestrator.TransformerAnomalyDetector"):
                    orchestrator = AIOpsOrchestrator(mock_config)
                    
                    # Mock strategy registry
                    mock_strategy = MagicMock()
                    orchestrator.remediation_registry.get_strategy = MagicMock(return_value=[mock_strategy])
                    
                    orchestrator._remediate_anomalies({"high_error_rate": True})
                    
                    mock_strategy.execute.assert_called_once_with(orchestrator, True)

def test_run_orchestrator(mock_config):
    with patch("src.aiops.aiops_orchestrator.PrometheusClient") as MockProm:
        with patch("src.aiops.aiops_orchestrator.IsolationForestDetector"):
            with patch("src.aiops.aiops_orchestrator.AutoencoderDetector"):
                with patch("src.aiops.aiops_orchestrator.DataDriftDetector"):
                    with patch("src.aiops.aiops_orchestrator.TransformerAnomalyDetector"):
                        with patch("src.aiops.aiops_orchestrator.push_metrics"):
                            mock_prom_instance = MockProm.return_value
                            mock_prom_instance.get_5xx_error_rate.return_value = 0.0
                            mock_prom_instance.get_p95_latency.return_value = 0.0
                            
                            orchestrator = AIOpsOrchestrator(mock_config)
                            
                            # Run 1 iteration
                            orchestrator.run(iterations=1)
                            
                            mock_prom_instance.get_5xx_error_rate.assert_called()
