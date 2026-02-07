from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ml.trainer import InstrumentedTrainer, PyTorchTrainer


@pytest.fixture
def dummy_data():
    X = np.random.rand(100, 5)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    return X, y

@patch("src.ml.trainer.mlflow")
@patch("src.ml.trainer.push_metrics")
def test_xgboost_trainer(mock_push, mock_mlflow, dummy_data):
    X, y = dummy_data
    trainer = InstrumentedTrainer(study_name="test_xgboost")
    params = {"framework": "xgboost", "n_estimators": 10, "max_depth": 3}
    
    acc = trainer.train_and_evaluate(X, y, params)
    assert 0 <= acc <= 1.0
    assert trainer.model is not None
    mock_mlflow.start_run.assert_called()

@patch("src.ml.trainer.mlflow")
@patch("src.ml.trainer.push_metrics")
def test_sklearn_trainer(mock_push, mock_mlflow, dummy_data):
    X, y = dummy_data
    trainer = InstrumentedTrainer(study_name="test_sklearn")
    params = {"framework": "sklearn", "n_estimators": 5}
    
    acc = trainer.train_and_evaluate(X, y, params)
    assert 0 <= acc <= 1.0
    assert trainer.model is not None

@patch("src.ml.trainer.mlflow")
@patch("src.ml.trainer.push_metrics")
def test_pytorch_trainer(mock_push, mock_mlflow, dummy_data):
    X, y = dummy_data
    trainer = InstrumentedTrainer(study_name="test_pytorch")
    params = {"framework": "pytorch", "epochs": 2, "lr": 0.01}
    
    acc = trainer.train_and_evaluate(X, y, params)
    assert 0 <= acc <= 1.0
    assert trainer.model is not None

@patch("src.ml.trainer.optuna.create_study")
def test_optimize(mock_create_study):
    mock_study = MagicMock()
    mock_study.best_params = {"a": 1}
    mock_study.best_value = 0.9
    mock_create_study.return_value = mock_study
    
    trainer = InstrumentedTrainer(study_name="test_opt")
    objective = MagicMock()
    study = trainer.optimize(objective, n_trials=1)
    
    assert study == mock_study
    assert trainer.best_params == {"a": 1}
    mock_create_study.assert_called_once()

def test_pytorch_predict_numpy(dummy_data):
    X, y = dummy_data
    trainer = PyTorchTrainer()
    model = trainer.train(X, y, X, y, {"epochs": 1})
    preds = trainer.predict(model, X)
    assert preds.shape == (100,)
    assert set(preds).issubset({0, 1})

def test_feature_importance_plot():
    trainer = InstrumentedTrainer(study_name="test_plot")
    importance = {"feat1": 0.5, "feat2": 0.3}
    with patch("src.ml.trainer.plt.savefig") as mock_save:
        path = trainer._plot_feature_importance(importance, "test_fw")
        assert "feature_importance.png" in path
        mock_save.assert_called()
