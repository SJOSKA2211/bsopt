import unittest

import numpy as np

from services.ml.indicators import get_adx, get_atr, get_bbands, get_ema, get_macd, get_rsi


class TestIndicators(unittest.TestCase):
    def setUp(self):
        self.prices = np.array([100, 101, 102, 101, 100, 99, 98, 99, 100, 101], dtype=float)
        self.high = self.prices + 1
        self.low = self.prices - 1
        self.close = self.prices

    def test_get_ema(self):
        ema = get_ema(self.prices, span=5)
        self.assertEqual(len(ema), len(self.prices))
        self.assertFalse(np.isnan(ema[0]))
        self.assertAlmostEqual(ema[0], self.prices[0])

    def test_get_rsi(self):
        rsi = get_rsi(self.prices, length=5)
        self.assertEqual(len(rsi), len(self.prices))
        self.assertTrue(np.isnan(rsi[0]))
        self.assertFalse(np.isnan(rsi[-1]))

    def test_get_bbands(self):
        lower, mid, upper = get_bbands(self.prices, length=5)
        self.assertEqual(len(lower), len(self.prices))
        # Filter out NaNs for comparison
        valid = ~np.isnan(mid)
        self.assertTrue(np.all(upper[valid] >= mid[valid]))
        self.assertTrue(np.all(mid[valid] >= lower[valid]))

    def test_get_macd(self):
        macd, signal, hist = get_macd(self.prices, fast=3, slow=6, signal=3)
        self.assertEqual(len(macd), len(self.prices))

    def test_get_atr(self):
        atr = get_atr(self.high, self.low, self.close, length=5)
        self.assertEqual(len(atr), len(self.prices))

    def test_get_adx(self):
        adx = get_adx(self.high, self.low, self.close, length=5)
        self.assertEqual(len(adx), len(self.prices))


if __name__ == "__main__":
    unittest.main()
