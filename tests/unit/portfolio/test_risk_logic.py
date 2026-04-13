import pytest
from src.portfolio.risk import RiskAttributor, PnLExplainer

def test_risk_attributor_aggregation():
    portfolio = [
        {"symbol": "AAPL", "quantity": 100, "type": "STOCK"}, # delta=1.0 default
        {"symbol": "OPT1", "quantity": 10, "type": "CALL", "delta": 0.5, "gamma": 0.01, "vega": 5, "theta": -2},
        {"symbol": "OPT2", "quantity": -5, "type": "PUT", "delta": -0.3, "gamma": 0.02, "vega": 10, "theta": -1},
    ]
    
    attributor = RiskAttributor(portfolio)
    totals = attributor.aggregate_greeks()
    
    # Delta: (100*1.0) + (10*0.5) + (-5*-0.3) = 100 + 5 + 1.5 = 106.5
    assert totals["delta"] == 106.5
    # Gamma: (100*0) + (10*0.01) + (-5*0.02) = 0.1 - 0.1 = 0
    assert totals["gamma"] == 0.0
    # Vega: (100*0) + (10*5) + (-5*10) = 50 - 50 = 0
    assert totals["vega"] == 0.0
    # Theta: (100*0) + (10*-2) + (-5*-1) = -20 + 5 = -15
    assert totals["theta"] == -15.0

def test_risk_stress_test():
    portfolio = [{"quantity": 100, "delta": 0.5, "gamma": 0.02, "vega": 10}]
    attributor = RiskAttributor(portfolio)
    
    # Delta=50, Gamma=2, Vega=1000
    res = attributor.run_stress_test(spot_move=2.0, vol_move=0.1)
    
    # Delta impact: 50 * 2.0 = 100
    # Gamma impact: 0.5 * 2 * (2^2) = 1 * 4 = 4
    # Vega impact: 1000 * 0.1 = 100
    # Total: 100 + 4 + 100 = 204
    assert res["total_pnl"] == 204.0
    assert res["delta_impact"] == 100.0
    assert res["gamma_impact"] == 4.0
    assert res["vega_impact"] == 100.0

def test_pnl_explainer():
    start_greeks = {"delta": 50, "gamma": 2, "vega": 1000, "theta": -10}
    end_greeks = {"delta": 55} # end_greeks not used in current implementation of explain_pnl
    
    explainer = PnLExplainer(start_greeks, end_greeks)
    res = explainer.explain_pnl(spot_move=1.0, vol_move=0.05, dt=1.0/365.0)
    
    # Delta: 50 * 1 = 50
    # Gamma: 0.5 * 2 * 1^2 = 1
    # Vega: 1000 * 0.05 = 50
    # Theta: -10 * (1/365) = -0.027397
    assert res["delta_pnl"] == 50.0
    assert res["gamma_pnl"] == 1.0
    assert res["vega_pnl"] == 50.0
    assert np.isclose(res["theta_pnl"], -0.027397, atol=1e-5)
    assert np.isclose(res["explained_total"], 100.972602, atol=1e-5)

import numpy as np # Needed for isclose