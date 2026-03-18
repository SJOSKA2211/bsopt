import numpy as np
import pandas as pd

from src.ml.feature_store.features import LogReturnFeature, SyntheticOHLCFeature
from src.ml.feature_store.store import feature_store


def test_log_return_feature():
    data = pd.DataFrame({"close": [100, 105, 102]})
    feature = LogReturnFeature()
    result = feature.transform(data)

    assert len(result) == 3
    assert result.iloc[0] == 0  # First element is NaN -> fillna(0)
    assert np.isclose(result.iloc[1], np.log(105 / 100))
    assert np.isclose(result.iloc[2], np.log(102 / 105))


def test_synthetic_ohlc_feature():
    data = pd.DataFrame({"price": [100, 101], "volume": [10, 20]})
    feature = SyntheticOHLCFeature()
    result = feature.transform(data)

    assert "open" in result.columns
    assert "high" in result.columns
    assert "low" in result.columns
    assert "close" in result.columns
    assert result.iloc[0]["open"] == 100
    assert result.iloc[1]["close"] == 101


def test_feature_store_compute():
    data = pd.DataFrame({"close": [100, 110, 121], "volume": [1, 2, 3]})
    result = feature_store.compute_features(data, ["log_return"])

    assert "log_return" in result.columns
    assert np.isclose(result["log_return"].iloc[1], np.log(110 / 100))


def test_feature_store_synthetic_integration():
    data = pd.DataFrame({"price": [50, 55], "volume": [100, 200]})
    # "synthetic_ohlc" is applied implicitly in our simplified implementation
    # but we can also ask for features that depend on it if we had any.
    # Here we just check if it returns the dataframe.
    # Ideally, compute_features should return the transformed df.

    result = feature_store.compute_features(data, [])

    assert "open" in result.columns
    assert result.iloc[0]["open"] == 50
