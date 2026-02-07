import pytest
import numpy as np
from src.pricing import quant_utils

def test_corrado_miller():
    S = np.array([100.0])
    K = np.array([100.0])
    T = np.array([1.0])
    r = np.array([0.05])
    q = np.array([0.0])
    price = np.array([10.45])
    option_type = np.array([0])
    
    iv = quant_utils.corrado_miller_initial_guess(price, S, K, T, r, q, option_type)
    assert 0.15 < iv[0] < 0.25

def test_vectorized_bs():
    S = np.array([100.0, 110.0])
    K = np.array([100.0, 100.0])
    T = np.array([1.0, 1.0])
    sigma = np.array([0.2, 0.2])
    r = np.array([0.05, 0.05])
    q = np.array([0.0, 0.0])
    is_call = np.array([True, True])
    
    prices = quant_utils.batch_bs_price_jit(S, K, T, sigma, r, q, is_call)
    assert prices[1] > prices[0] # ITM should be more expensive
