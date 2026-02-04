import numpy as np
import pytest
from src.pricing.calibration.engine import HestonCalibrator, MarketOption

def test_market_option_spread():
    opt = MarketOption(T=1.0, strike=100, spot=100, price=10, bid=9, ask=11, volume=100, open_interest=100, option_type='call')
    assert opt.spread == 2.0

def test_heston_calibrator_insufficient_liquid():
    calibrator = HestonCalibrator()
    # Provide empty data to hit line 100
    with pytest.raises(ValueError, match="Insufficient liquid options"):
        calibrator.calibrate([])

def test_weighted_objective_feller_penalty():
    calibrator = HestonCalibrator()
    # kappa=1, theta=0.01, sigma=1.0. 2*kappa*theta = 0.02. sigma^2 = 1.0.
    # feller violated.
    params = np.array([1.0, 0.01, 1.0, 0.0, 0.04])
    res = calibrator._weighted_objective(params, [])
    assert res == 1e12

def test_weighted_objective_put():
    calibrator = HestonCalibrator()
    opt = MarketOption(T=1.0, strike=100, spot=100, price=10, bid=9, ask=11, volume=100, open_interest=100, option_type='put')
    # Use valid params to hit line 80
    params = np.array([2.0, 0.04, 0.1, 0.0, 0.04])
    res = calibrator._weighted_objective(params, [opt])
    assert res < 1e12

def test_weighted_objective_exception_continue():
    calibrator = HestonCalibrator()
    opt = MarketOption(T=1.0, strike=100, spot=100, price=10, bid=9, ask=11, volume=100, open_interest=100, option_type='call')
    # Force exception in loop by mocking HestonModelFFT
    from unittest.mock import patch
    params = np.array([2.0, 0.04, 0.1, 0.0, 0.04])
    with patch('src.pricing.calibration.engine.HestonModelFFT', side_effect=Exception("mock fail")):
        res = calibrator._weighted_objective(params, [opt])
        # If all items failed, total_weight is 0, returns 1e12
        assert res == 1e12
