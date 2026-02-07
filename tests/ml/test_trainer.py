from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ml.trainer import InstrumentedTrainer


@pytest.fixture
def sample_data():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture(autouse=True)
def mock_mlflow():
    with patch("mlflow.start_run"), patch("mlflow.log_params"), patch(
        "mlflow.log_metric"
    ), patch("mlflow.set_tracking_uri"):
        yield


def test_trainer_initialization():
    trainer = InstrumentedTrainer(study_name="test_study")
    assert trainer.study_name == "test_study"
    assert trainer.model is None


def test_trainer_train_xgboost(sample_data):
    X, y = sample_data
    trainer = InstrumentedTrainer(study_name="test_study")

    # Define a simple search space for Optuna
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 20),
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "framework": "xgboost",
        }
        return trainer.train_and_evaluate(X, y, params)

    study = trainer.optimize(objective, n_trials=2)

    assert len(study.trials) == 2
    assert trainer.model is not None


@patch("src.ml.trainer.logger")
def test_trainer_logging(mock_logger, sample_data):
    X, y = sample_data
    trainer = InstrumentedTrainer(study_name="test_study")
    params = {"max_depth": 3, "learning_rate": 0.1}

    trainer.train_and_evaluate(X, y, params)

    assert mock_logger.info.called


def test_trainer_prometheus_metrics_usage(sample_data):
    """Verify that Prometheus metrics are updated during training."""
    from src.shared import observability

    X, y = sample_data
    trainer = InstrumentedTrainer(study_name="test_study")

    # Mock the metrics to verify they are called
    with patch.object(
        observability.TRAINING_DURATION, "labels"
    ) as mock_duration_labels, patch.object(
        observability.MODEL_ACCURACY, "labels"
    ) as mock_accuracy_labels:

        mock_duration_observe = MagicMock()
        mock_duration_labels.return_value.observe = mock_duration_observe

        mock_accuracy_set = MagicMock()
        mock_accuracy_labels.return_value.set = mock_accuracy_set

        params = {"max_depth": 3, "learning_rate": 0.1, "framework": "xgboost"}
        trainer.train_and_evaluate(X, y, params)

        mock_duration_labels.assert_called_with(framework="xgboost")
        mock_duration_observe.assert_called_once()

        mock_accuracy_labels.assert_called_with(framework="xgboost")
        mock_accuracy_set.assert_called_once()


def test_trainer_mlflow_uri():
    with patch("mlflow.set_tracking_uri") as mock_set_uri:
        InstrumentedTrainer(
            study_name="test_study", tracking_uri="http://localhost:5000"
        )
        mock_set_uri.assert_called_once_with("http://localhost:5000")


@patch("mlflow.log_params")
@patch("mlflow.log_metric")
def test_mlflow_logging_content(mock_log_metric, mock_log_params, sample_data):
    X, y = sample_data
    trainer = InstrumentedTrainer(study_name="test_study")
    params = {"max_depth": 3, "learning_rate": 0.1}

    trainer.train_and_evaluate(X, y, params)

    mock_log_params.assert_called_with(params)
    # Check if accuracy was logged
    metric_names = [call.args[0] for call in mock_log_metric.call_args_list]
    assert "accuracy" in metric_names
    assert "duration" in metric_names


def test_trainer_frameworks(sample_data):
    """Verify that multiple frameworks can be used for training."""
    X, y = sample_data
    trainer = InstrumentedTrainer(study_name="multi_framework")

    # Test Scikit-learn (RandomForest)
    params_sk = {"n_estimators": 10, "max_depth": 3, "framework": "sklearn"}
    acc_sk = trainer.train_and_evaluate(X, y, params_sk)
    assert acc_sk >= 0

    # Test PyTorch (Simple Neural Net)
    params_torch = {"epochs": 2, "lr": 0.01, "framework": "pytorch"}
    acc_torch = trainer.train_and_evaluate(X, y, params_torch)
    assert acc_torch >= 0


@patch("src.ml.trainer.train_xgboost_distributed")
def test_trainer_dask_xgboost(mock_train_dist, sample_data):
    """Verify that DaskXGBoostTrainer correctly calls the distributed training utility."""
    X, y = sample_data
    trainer = InstrumentedTrainer(study_name="dask_test")

    mock_booster = MagicMock()
    mock_train_dist.return_value = mock_booster

    params = {
        "n_estimators": 10,
        "framework": "dask_xgboost",
        "dask_address": "localhost:8786",
    }
    trainer.train_and_evaluate(X, y, params)

    assert mock_train_dist.called
    args, kwargs = mock_train_dist.call_args
    assert kwargs["dask_address"] == "localhost:8786"
    assert trainer.model == mock_booster


def test_trainer_optuna_integration(sample_data):
    """Verify that Optuna correctly optimizes hyperparameters."""
    X, y = sample_data
    trainer = InstrumentedTrainer(study_name="optuna_test")

    def objective(trial):
        # Create a simple dependency on n_estimators to see if Optuna moves towards it
        n_estimators = trial.suggest_int("n_estimators", 10, 50)
        params = {"n_estimators": n_estimators, "max_depth": 3, "framework": "xgboost"}
        return trainer.train_and_evaluate(X, y, params)

    study = trainer.optimize(objective, n_trials=5)

    assert len(study.trials) == 5
    assert "n_estimators" in study.best_params
    assert study.best_value >= 0


def test_base_trainer_abstract_methods():
    """Verify that BaseTrainer raises NotImplementedError or returns default values."""
    from src.ml.trainer import BaseTrainer

    trainer = BaseTrainer()

    with pytest.raises(NotImplementedError):
        trainer.train(None, None, None, None, {})

    with pytest.raises(NotImplementedError):
        trainer.predict(None, None)

    # These should not raise but just pass/return None
    trainer.log_model(None, "test")
    assert trainer.get_feature_importance(None, []) is None


@patch("mlflow.log_artifact")
@patch("mlflow.sklearn.log_model")
def test_trainer_mlflow_artifacts(mock_log_model, mock_log_artifact, sample_data):
    """Verify that models and artifacts are logged to MLflow."""
    X, y = sample_data
    trainer = InstrumentedTrainer(study_name="artifact_test")

    params = {"n_estimators": 5, "framework": "sklearn"}
    trainer.train_and_evaluate(X, y, params)

    # Verify model was logged
    assert mock_log_model.called
    # Verify artifact (e.g., feature importance plot) was logged
    # We will implement this in the trainer
    assert mock_log_artifact.called


@patch("mlflow.set_tags")
def test_trainer_dataset_lineage(mock_set_tags, sample_data):
    """Verify that dataset metadata is logged as tags."""
    X, y = sample_data
    trainer = InstrumentedTrainer(study_name="lineage_test")

    params = {"n_estimators": 5, "framework": "xgboost"}
    metadata = {"dataset_version": "v1.0", "source": "test_source"}

    trainer.train_and_evaluate(X, y, params, dataset_metadata=metadata)

    mock_set_tags.assert_called_with(metadata)
