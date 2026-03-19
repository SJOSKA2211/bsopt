import numpy as np
import structlog
from typing import Dict, List, Any

logger = structlog.get_logger(__name__)

class RiskAttributor:
    """
    Institutional Greeks Risk Attributor.
    Aggregates exposure and runs stress tests for the EquaFlow platform.
    """
    def __init__(self, portfolio_data: List[Dict[str, Any]]):
        self.portfolio = portfolio_data

    def aggregate_greeks(self) -> Dict[str, float]:
        """Aggregate Delta, Gamma, Vega, Theta across all positions."""
        totals = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        for pos in self.portfolio:
            qty = pos.get("quantity", 0)
            totals["delta"] += pos.get("delta", 0.0) * qty
            totals["gamma"] += pos.get("gamma", 0.0) * qty
            totals["vega"] += pos.get("vega", 0.0) * qty
            totals["theta"] += pos.get("theta", 0.0) * qty
            
        logger.info("greeks_aggregated", totals=totals)
        return totals

    def run_stress_test(self, spot_move: float, vol_move: float) -> Dict[str, float]:
        """
        Estimate P&L impact under a defined stress scenario.
        Using a Taylor Series expansion (Greeks-based).
        """
        greeks = self.aggregate_greeks()
        
        # dP = Delta * dS + 0.5 * Gamma * dS^2 + Vega * dV + Theta * dt
        # For simplicity, we assume dS is fixed move, dt=0
        delta_pnl = greeks["delta"] * spot_move
        gamma_pnl = 0.5 * greeks["gamma"] * (spot_move ** 2)
        vega_pnl = greeks["vega"] * vol_move
        
        total_pnl = delta_pnl + gamma_pnl + vega_pnl
        
        logger.warning("stress_test_audit", 
                       spot_move=spot_move, 
                       vol_move=vol_move, 
                       total_pnl=total_pnl)
        
        return {
            "total_pnl": total_pnl,
            "delta_impact": delta_pnl,
            "gamma_impact": gamma_pnl,
            "vega_impact": vega_pnl
        }

if __name__ == "__main__":
    # Mock data for demonstration
    mock_portfolio = [
        {"symbol": "NIFTY_CALL_25000", "quantity": 100, "delta": 0.5, "gamma": 0.02, "vega": 10.0, "theta": -5.0},
        {"symbol": "NIFTY_PUT_24000", "quantity": -50, "delta": -0.3, "gamma": 0.01, "vega": 8.0, "theta": -3.0}
    ]
    attributor = RiskAttributor(mock_portfolio)
    print(attributor.run_stress_test(spot_move=100, vol_move=0.05))
