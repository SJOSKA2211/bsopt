import pytest
import numpy as np
from src.pricing.calibration.engine import MarketOption, HestonCalibrator
from src.pricing.models.heston_fft import HestonParams

@pytest.fixture
def sample_options():
    options = []
    # Create 10 options to pass MIN_LIQUID_OPTIONS=5
    for strike in [90, 95, 100, 105, 110, 90, 95, 100, 105, 110]:
        options.append(MarketOption(
            T=1.0, strike=strike, spot=100.0, price=10.0,
            bid=9.5, ask=10.5, volume=100, open_interest=50,
            option_type="call"
        ))
    return options

def test_market_option_properties():
    opt = MarketOption(
        T=1.0, strike=100.0, spot=100.0, price=10.0,
        bid=9.0, ask=11.0, volume=100, open_interest=50,
        option_type="call"
    )
    assert opt.mid_price == 10.0
    assert opt.spread == 2.0
    assert opt.moneyness == 1.0

def test_heston_calibrator_filter_liquid(sample_options):
    calibrator = HestonCalibrator()
    # Add an illiquid option
    sample_options.append(MarketOption(
        T=1.0, strike=100.0, spot=100.0, price=10.0,
        bid=9.5, ask=10.5, volume=0, open_interest=0,
        option_type="call"
    ))
    filtered = calibrator._filter_liquid_options(sample_options)
    assert len(filtered) == 10 # Original 10 are liquid

def test_heston_calibrator_weighted_objective(sample_options):
    calibrator = HestonCalibrator()
    liquid = calibrator._filter_liquid_options(sample_options)
    
    # kappa, theta, sigma, rho, v0
    params = np.array([2.0, 0.04, 0.1, -0.7, 0.04])
    obj = calibrator._weighted_objective(params, liquid)
    assert obj >= 0
    
    # Feller violation penalty
    invalid_params = np.array([0.1, 0.04, 1.0, -0.7, 0.04])
    assert calibrator._weighted_objective(invalid_params, liquid) == 1e12

def test_heston_calibrator_calibrate(sample_options):
    calibrator = HestonCalibrator()
    # Mocking price_call/put might be too complex, let's run a small calibration
    final_params, metrics = calibrator.calibrate(sample_options, maxiter=1, popsize=5)
    assert isinstance(final_params, HestonParams)
    assert "rmse" in metrics
    assert metrics["num_options"] == 10

def test_heston_calibrator_calibrate_insufficient_options():
    calibrator = HestonCalibrator()
    with pytest.raises(ValueError, match="Insufficient liquid options"):
        calibrator.calibrate([])

def test_heston_calibrator_calibrate_put(sample_options):
    # Convert options to puts
    for opt in sample_options:
        opt.option_type = "put"
    calibrator = HestonCalibrator()
    # Trigger weighted_objective with puts
    params = np.array([2.0, 0.04, 0.1, -0.7, 0.04])
    obj = calibrator._weighted_objective(params, sample_options[:5])
    assert obj >= 0

def test_calibrate_surface(sample_options, mocker):
    # Need to mock SVISurface and implied_volatility
    mocker.patch("src.pricing.calibration.svi_surface.SVISurface.fit_svi_slice", return_value=(0.1, 0.1, 0.0, 0.0, 0.1))
    mocker.patch("src.pricing.implied_vol.implied_volatility", return_value=0.2)
    
    calibrator = HestonCalibrator()
    # Split sample_options into two maturities
    for i in range(5):
        sample_options[i].T = 0.5
        
    surface_params = calibrator.calibrate_surface(sample_options)
    assert 0.5 in surface_params
    assert 1.0 in surface_params

def test_weighted_objective_all_fail(sample_options, mocker):
    calibrator = HestonCalibrator()
    # Mock HestonModelFFT to always fail
    mocker.patch("src.pricing.calibration.engine.HestonModelFFT", side_effect=Exception("Fail"))
    params = np.array([2.0, 0.04, 0.1, -0.7, 0.04])
    assert calibrator._weighted_objective(params, sample_options) == 1e12

def test_calibrate_surface_insufficient_options_per_maturity(mocker):
    # Maturity 2.0 has 4 options < MIN_LIQUID_OPTIONS=5
    calibrator = HestonCalibrator()
    options = []
    for _ in range(4):
        options.append(MarketOption(T=2.0, strike=100, spot=100, price=10, bid=9, ask=11, volume=100, open_interest=50, option_type="call"))
    
    # Also add 5 options for maturity 1.0 to ensure it doesn't return empty early
    for _ in range(5):
        options.append(MarketOption(T=1.0, strike=100, spot=100, price=10, bid=9, ask=11, volume=100, open_interest=50, option_type="call"))
        
    mocker.patch("src.pricing.implied_vol.implied_volatility", return_value=0.2)
    mocker.patch("src.pricing.calibration.svi_surface.SVISurface.fit_svi_slice", return_value=(0.1, 0.1, 0.0, 0.0, 0.1))
    
    res = calibrator.calibrate_surface(options)
    assert 2.0 not in res
    assert 1.0 in res

def test_calibrate_surface_insufficient_ivs(sample_options, mocker):
    # Case where many IVs fail, resulting in < MIN_LIQUID_OPTIONS valid log_strikes
    mocker.patch("src.pricing.implied_vol.implied_volatility", side_effect=Exception("All fail"))
    calibrator = HestonCalibrator()
    res = calibrator.calibrate_surface(sample_options)
    assert len(res) == 0
