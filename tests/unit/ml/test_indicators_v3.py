import numpy as np

from src.ml import indicators


def test_numba_ema():
    values = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=np.float64)
    res = indicators._numba_ema(values, 3)
    assert len(res) == 5
    assert not np.isnan(res[-1])


def test_numba_rsi():
    prices = np.array(
        [10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 8.0, 9.0, 10.0, 11.0], dtype=np.float64
    )
    res = indicators._numba_rsi(prices, 3)
    assert len(res) == 10
    assert not np.isnan(res[-1])


def test_numba_bbands():
    prices = np.array([100.0, 101.0, 102.0, 101.0, 100.0], dtype=np.float64)
    # Order: lower, mid, upper
    lower, mid, upper = indicators._numba_bbands(prices, 3, 2.0)
    mask = ~np.isnan(upper)
    assert any(mask)
    assert all(upper[mask] >= mid[mask])
    assert all(mid[mask] >= lower[mask])


def test_numba_macd():
    prices = np.random.rand(100).astype(np.float64)
    macd, signal, hist = indicators._numba_macd(prices, 12, 26, 9)
    assert len(macd) == 100


def test_numba_atr():
    high = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    low = np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float64)
    close = np.array([9.5, 10.5, 11.5, 12.5], dtype=np.float64)
    res = indicators._numba_atr(high, low, close, 2)
    assert len(res) == 4
