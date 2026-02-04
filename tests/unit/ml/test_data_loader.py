import pytest
from src.ml.data_loader import DataNormalizer

def test_normalize_incomplete_scraper_data():
    raw_data = {
        "timestamp": "2026-01-14T10:00:00",
        "price": 15.50,
        "volume": 1000,
        "symbol": "SCOM"
    }
    normalized = DataNormalizer.normalize_incoming(raw_data)
    
    assert normalized["open"] == 15.50
    assert normalized["high"] == 15.50
    assert normalized["low"] == 15.50
    assert normalized["close"] == 15.50
    assert normalized["volume"] == 1000
    assert normalized["source_type"] == "scraper_synthetic"

def test_normalize_complete_data():
    raw_data = {
        "timestamp": "2026-01-14T10:00:00",
        "open": 15.0,
        "high": 16.0,
        "low": 14.0,
        "close": 15.5,
        "volume": 1000
    }
    normalized = DataNormalizer.normalize_incoming(raw_data)
    
    assert normalized["open"] == 15.0
    assert normalized["close"] == 15.5
    assert "source_type" not in normalized

def test_outlier_removal_true():
    data = {"close": 115.0, "symbol": "AAPL"}
    prev_price = 100.0
    # 15% change > 10% threshold
    assert DataNormalizer.remove_outliers(data, prev_price, threshold=0.1) is True

def test_outlier_removal_false():
    data = {"close": 105.0, "symbol": "AAPL"}
    prev_price = 100.0
    # 5% change < 10% threshold
    assert DataNormalizer.remove_outliers(data, prev_price, threshold=0.1) is False

def test_outlier_removal_edge_cases():
    assert DataNormalizer.remove_outliers({}, 0) is False
    assert DataNormalizer.remove_outliers({"close": 100}, 0) is False
    assert DataNormalizer.remove_outliers({}, 100) is False
    
def test_outlier_removal_price_key():
    data = {"price": 115.0, "symbol": "AAPL"}
    prev_price = 100.0
    assert DataNormalizer.remove_outliers(data, prev_price, threshold=0.1) is True