import pytest
import numpy as np
import pandas as pd
from src.portfolio.engine import PortfolioOptimizer, RebalancingEngine

@pytest.fixture
def sample_data():
    symbols = ["AAPL", "GOOGL", "MSFT"]
    dates = pd.date_range("2023-01-01", periods=10)
    # Generate random returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, (10, 3))
    df = pd.DataFrame(returns, columns=symbols, index=dates)
    return symbols, df

def test_portfolio_optimizer_init(sample_data):
    symbols, df = sample_data
    optimizer = PortfolioOptimizer(symbols, df)
    assert optimizer.symbols == symbols
    assert optimizer.returns.shape == (10, 3)
    assert optimizer.cov_matrix.shape == (3, 3)
    assert len(optimizer.mean_returns) == 3

def test_optimize_hrp(sample_data):
    symbols, df = sample_data
    optimizer = PortfolioOptimizer(symbols, df)
    weights = optimizer.optimize_weights(method="hrp")
    assert len(weights) == 3
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0)

def test_optimize_mvo(sample_data):
    symbols, df = sample_data
    optimizer = PortfolioOptimizer(symbols, df)
    weights = optimizer.optimize_weights(method="mvo")
    assert len(weights) == 3
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= -1e-7)  # Allow small numerical debt

def test_optimize_black_litterman(sample_data):
    symbols, df = sample_data
    optimizer = PortfolioOptimizer(symbols, df)
    views = np.array([0.02, 0.01, 0.03])
    confidences = np.array([0.1, 0.1, 0.1])
    weights = optimizer.optimize_weights(
        method="black_litterman", 
        views=views, 
        confidences=confidences
    )
    assert len(weights) == 3
    assert np.isclose(weights.sum(), 1.0)

def test_optimizer_invalid_method(sample_data):
    symbols, df = sample_data
    optimizer = PortfolioOptimizer(symbols, df)
    with pytest.raises(ValueError, match="Unknown optimization method"):
        optimizer.optimize_weights(method="invalid")

def test_rebalancing_engine_calculation():
    current_positions = {"AAPL": 10.0, "GOOGL": 5.0}
    prices = {"AAPL": 150.0, "GOOGL": 2800.0}
    target_weights = {"AAPL": 0.4, "GOOGL": 0.6}
    total_nav = 20000.0  # (10*150 + 5*2800) = 1500 + 14000 = 15500
    
    engine = RebalancingEngine(current_positions, prices)
    orders = engine.calculate_rebalance(target_weights, total_nav)
    
    # Target AAPL = 20000 * 0.4 = 8000. Current AAPL = 1500. Diff = 6500. Shares = 6500 / 150 = 43.33
    # Target GOOGL = 20000 * 0.6 = 12000. Current GOOGL = 14000. Diff = -2000. Shares = -2000 / 2800 = -0.714
    
    assert orders["AAPL"] == (8000 - 1500) / 150.0
    assert orders["GOOGL"] == (12000 - 14000) / 2800.0