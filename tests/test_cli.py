import sys
from unittest.mock import MagicMock

from click.testing import CliRunner

# Mock imports before loading bs_cli
sys.modules["scripts.enforce_venv"] = MagicMock()
sys.modules["src.services.pricing_service"] = MagicMock()
sys.modules["src.ml.reinforcement_learning.train"] = MagicMock()
sys.modules["src.shared.shm_manager"] = MagicMock()

from bs_cli import cli  # noqa: E402


class TestCli:
    def test_price_command(self):
        runner = CliRunner()
        
        # Mock PricingService
        mock_service_cls = sys.modules["src.services.pricing_service"].PricingService
        mock_service_instance = mock_service_cls.return_value
        
        mock_result = MagicMock()
        mock_result.price = 10.5
        mock_result.computation_time_ms = 1.2
        mock_result.model = "black_scholes"
        
        # Setup async return
        async def mock_price_option(*args, **kwargs):
            return mock_result
        mock_service_instance.price_option.side_effect = mock_price_option

        result = runner.invoke(cli, [
            'price', 
            '--spot', '100', 
            '--strike', '100', 
            '--maturity', '1', 
            '--volatility', '0.2', 
            '--rate', '0.05'
        ])
        
        assert result.exit_code == 0
        assert "Pricing Result: BLACK_SCHOLES" in result.output
        assert "$10.5000" in result.output

    def test_train_transformer_command(self):
        runner = CliRunner()
        
        mock_train = sys.modules["src.ml.reinforcement_learning.train"].train_td3
        mock_train.return_value = {'run_id': 'test-run', 'model_path': '/tmp/model'}
        
        result = runner.invoke(cli, ['train_transformer', '--timesteps', '100'])
        
        assert result.exit_code == 0
        assert "Training Complete!" in result.output
        assert "test-run" in result.output
        mock_train.assert_called_with(total_timesteps=100)

    def test_mesh_status_command(self):
        runner = CliRunner()
        
        mock_shm_cls = sys.modules["src.shared.shm_manager"].SHMManager
        mock_shm = mock_shm_cls.return_value
        mock_shm.name = "market_mesh"
        mock_shm.read.return_value = {"AAPL": 100}
        
        result = runner.invoke(cli, ['mesh_status'])
        
        assert result.exit_code == 0
        assert "Market Mesh SHM" in result.output
        assert "Active" in result.output
        assert "Tickers Tracked: 1" in result.output

    def test_mesh_status_offline(self):
        runner = CliRunner()
        
        mock_shm_cls = sys.modules["src.shared.shm_manager"].SHMManager
        mock_shm_cls.side_effect = Exception("SHM Error")
        
        result = runner.invoke(cli, ['mesh_status'])
        
        assert result.exit_code == 0
        assert "Market Mesh SHM" in result.output
        assert "Offline" in result.output
