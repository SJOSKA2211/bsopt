import sys
import pytest
from unittest.mock import MagicMock, patch

# Mock dependencies
sys.modules["mlflow"] = MagicMock()
sys.modules["mlflow.xgboost"] = MagicMock()
sys.modules["mlflow.sklearn"] = MagicMock()
sys.modules["mlflow.pytorch"] = MagicMock()
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()

from src.ml.tracker import ExperimentTracker

class TestTracker:
    def setUp(self):
        self.tracker = ExperimentTracker(study_name="test_study")

    @patch("src.ml.tracker.mlflow.start_run")
    def test_start_run(self, mock_start):
        self.tracker.start_run()
        assert mock_start.called

    @patch("src.ml.tracker.mlflow.log_params")
    def test_log_params(self, mock_log):
        self.tracker.log_params({"a": 1})
        assert mock_log.called

    @patch("src.ml.tracker.mlflow.log_metric")
    def test_log_metrics(self, mock_log):
        self.tracker.log_metrics(0.9, 0.1, 10.0, "xgboost")
        assert mock_log.called

    @patch("src.ml.tracker.plt.figure")
    @patch("src.ml.tracker.plt.savefig")
    @patch("src.ml.tracker.mlflow.log_artifact")
    @patch("src.ml.tracker.os.remove")
    @patch("src.ml.tracker.os.rmdir")
    def test_log_feature_importance(
        self, mock_rmdir, mock_remove, mock_artifact, mock_savefig, mock_fig
    ):
        importance = {"f1": 0.5, "f2": 0.3}
        self.tracker.log_feature_importance(importance, "xgboost")
        assert mock_artifact.called

