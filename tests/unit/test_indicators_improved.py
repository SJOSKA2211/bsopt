import numpy as np
import pytest

from src.ml.indicators import get_adx, get_atr, get_bbands, get_ema, get_macd, get_rsi


class TestIndicators:
    def setUp(self):
        self.prices = np.array([100, 101, 102, 101, 100, 99, 98, 99, 100, 101], dtype=float)
        self.high = self.prices + 1
        self.low = self.prices - 1
        self.close = self.prices

    def test_get_ema(self):
        ema = get_ema(self.prices, span=5)
        assert len(ema) == len(self.prices)
        assert not np.isnan(ema[0])
        assert ema[0] == pytest.approx(self.prices[0])

    def test_get_rsi(self):
        rsi = get_rsi(self.prices, length=5)
        assert len(rsi) == len(self.prices)
        assert np.isnan(rsi[0])
        assert not np.isnan(rsi[-1])

    def test_get_bbands(self):
        lower, mid, upper = get_bbands(self.prices, length=5)
        assert len(lower) == len(self.prices)
        # Filter out NaNs for comparison
        valid = ~np.isnan(mid)
        assert np.all(upper[valid] >= mid[valid])
        assert np.all(mid[valid] >= lower[valid])

    def test_get_macd(self):
        macd, signal, hist = get_macd(self.prices, fast=3, slow=6, signal=3)
        assert len(macd) == len(self.prices)

    def test_get_atr(self):
        atr = get_atr(self.high, self.low, self.close, length=5)
        assert len(atr) == len(self.prices)

    def test_get_adx(self):
        adx = get_adx(self.high, self.low, self.close, length=5)
        assert len(adx) == len(self.prices)

