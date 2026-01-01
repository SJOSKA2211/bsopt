import pytest
import numpy as np
import math
from src.pricing.quant_utils import fast_normal_cdf, fast_normal_pdf, corrado_miller_initial_guess

def test_fast_normal_cdf():
    assert pytest.approx(fast_normal_cdf(0), abs=1e-7) == 0.5
    assert pytest.approx(fast_normal_cdf(1.96), abs=1e-2) == 0.975

def test_fast_normal_pdf():
    assert pytest.approx(fast_normal_pdf(0), abs=1e-7) == 1.0 / math.sqrt(2 * math.pi)

def test_corrado_miller_guess():
    # S=100, K=100, T=1, r=0.05, q=0, sigma=0.2 -> price ~ 10.45
    market_prices = np.array([10.45])
    spots = np.array([100.0])
    strikes = np.array([100.0])
    maturities = np.array([1.0])
    rates = np.array([0.05])
    dividends = np.array([0.0])
    option_types = np.array([0]) # 0 for call
    
    sigma_guess = corrado_miller_initial_guess(
        market_prices, spots, strikes, maturities, rates, dividends, option_types
    )
    assert sigma_guess[0] > 0
    assert pytest.approx(sigma_guess[0], abs=0.05) == 0.2
