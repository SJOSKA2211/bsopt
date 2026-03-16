import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Mock xgboost and dask
sys.modules["xgboost"] = MagicMock()
sys.modules["dask"] = MagicMock()
sys.modules["dask.distributed"] = MagicMock()

from services.ml.strategies import (
    PyTorchStrategy,
    SklearnStrategy,
    XGBoostStrategy,
    get_strategy,
)


class TestStrategies(unittest.TestCase):
    def setUp(self):
        self.X_train = np.random.rand(100, 5)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.rand(20, 5)
        self.y_test = np.random.randint(0, 2, 20)

    def test_get_strategy(self):
        s = get_strategy("sklearn")
        self.assertIsInstance(s, SklearnStrategy)

    def test_sklearn_strategy(self):
        s = SklearnStrategy()
        model = s.train(self.X_train, self.y_train, self.X_test, self.y_test, {"n_estimators": 10})
        self.assertIsNotNone(model)
        preds = s.predict(model, self.X_test)
        self.assertEqual(len(preds), 20)

    def test_pytorch_strategy(self):
        s = PyTorchStrategy()
        model = s.train(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            {"epochs": 1, "lr": 0.01},
        )
        self.assertIsNotNone(model)
        preds = s.predict(model, self.X_test)
        self.assertEqual(len(preds), 20)

    @patch("services.ml.strategies.xgb.train")
    @patch("services.ml.strategies.xgb.DMatrix")
    def test_xgboost_strategy(self, mock_dmatrix, mock_train):
        mock_model = MagicMock()
        mock_model.best_iteration = 5
        mock_model.predict.return_value = np.random.rand(20)
        mock_train.return_value = mock_model

        s = XGBoostStrategy()
        model = s.train(self.X_train, self.y_train, self.X_test, self.y_test, {"n_estimators": 10})
        self.assertIsNotNone(model)
        preds = s.predict(model, self.X_test)
        self.assertEqual(len(preds), 20)


if __name__ == "__main__":
    unittest.main()
