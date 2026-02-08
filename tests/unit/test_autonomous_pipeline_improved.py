import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import sys

# Mock dependencies before imports
sys.modules["mlflow"] = MagicMock()
sys.modules["mlflow.pytorch"] = MagicMock()
sys.modules["optuna"] = MagicMock()

from src.ml.autonomous_pipeline import AutonomousMLPipeline

class TestAutonomousPipeline(unittest.TestCase):
    def setUp(self):
        self.config = {
            "api_key": "test_key",
            "db_url": "postgresql://user:pass@localhost/db",
            "ticker": "AAPL",
            "study_name": "test_study",
            "n_trials": 1,
            "framework": "xgboost"
        }
        with patch("src.ml.autonomous_pipeline.create_engine"), \
             patch("src.ml.autonomous_pipeline.Base.metadata.create_all"), \
             patch("src.ml.autonomous_pipeline.DriftTrigger"), \
             patch("src.ml.autonomous_pipeline.MarketDataScraper"):
            self.pipeline = AutonomousMLPipeline(self.config)

    def test_generate_features(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="D"),
            "close": np.random.randn(100).cumsum() + 100,
            "high": np.random.randn(100).cumsum() + 105,
            "low": np.random.randn(100).cumsum() + 95,
            "open": np.random.randn(100).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 100)
        })
        # Mock indicators
        with patch("src.ml.autonomous_pipeline.get_rsi", return_value=np.random.rand(100)), \
             patch("src.ml.autonomous_pipeline.get_macd", return_value=(np.random.rand(100), np.random.rand(100), np.random.rand(100))), \
             patch("src.ml.autonomous_pipeline.get_bbands", return_value=(np.random.rand(100), np.random.rand(100), np.random.rand(100))), \
             patch("src.ml.autonomous_pipeline.get_atr", return_value=np.random.rand(100)), \
             patch("src.ml.autonomous_pipeline.get_adx", return_value=np.random.rand(100)):
            df_feat = self.pipeline.generate_features(df)
            self.assertIn("RSI_14", df_feat.columns)
            self.assertIn("MACD_12_26_9", df_feat.columns)

    def test_prepare_training_data(self):
        df = pd.DataFrame({
            "close": np.random.rand(10),
            "feat1": np.random.rand(10)
        })
        x, y, names, meta = self.pipeline._prepare_training_data(df)
        self.assertEqual(len(x), 9)
        self.assertEqual(len(y), 9)
        self.assertIn("feat1", names)

if __name__ == '__main__':
    unittest.main()
