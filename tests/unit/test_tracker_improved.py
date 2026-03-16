import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock dependencies
sys.modules["mlflow"] = MagicMock()
sys.modules["mlflow.xgboost"] = MagicMock()
sys.modules["mlflow.sklearn"] = MagicMock()
sys.modules["mlflow.pytorch"] = MagicMock()
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()

from services.ml.tracker import ExperimentTracker


class TestTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = ExperimentTracker(study_name="test_study")

    @patch("services.ml.tracker.mlflow.start_run")
    def test_start_run(self, mock_start):
        self.tracker.start_run()
        self.assertTrue(mock_start.called)

    @patch("services.ml.tracker.mlflow.log_params")
    def test_log_params(self, mock_log):
        self.tracker.log_params({"a": 1})
        self.assertTrue(mock_log.called)

    @patch("services.ml.tracker.mlflow.log_metric")
    def test_log_metrics(self, mock_log):
        self.tracker.log_metrics(0.9, 0.1, 10.0, "xgboost")
        self.assertTrue(mock_log.called)

    @patch("services.ml.tracker.plt.figure")
    @patch("services.ml.tracker.plt.savefig")
    @patch("services.ml.tracker.mlflow.log_artifact")
    @patch("services.ml.tracker.os.remove")
    @patch("services.ml.tracker.os.rmdir")
    def test_log_feature_importance(
        self, mock_rmdir, mock_remove, mock_artifact, mock_savefig, mock_fig
    ):
        importance = {"f1": 0.5, "f2": 0.3}
        self.tracker.log_feature_importance(importance, "xgboost")
        self.assertTrue(mock_artifact.called)


if __name__ == "__main__":
    unittest.main()
