from typing import Any

import structlog

logger = structlog.get_logger(__name__)

class RiskAttributor:
    """
    Production Greeks Risk Attributor.
    Aggregates exposure and runs stress tests for the Manifold platform.
    """

    def __init__(self, portfolio_data: list[dict[str, Any]]):
        self.portfolio = portfolio_data

    def aggregate_greeks(self) -> dict[str, float]:
        """Aggregate Delta, Gamma, Vega, Theta across all positions (Multi-Asset aware)."""
        totals = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        for pos in self.portfolio:
            qty = pos.get("quantity", 0)

            # Options have specific Greeks; linear assets (stock/crypto) have Delta=1.0
            is_option = "type" in pos and pos["type"] in ["CALL", "PUT"]

            p_delta = pos.get("delta", 1.0 if not is_option else 0.0)
            p_gamma = pos.get("gamma", 0.0)
            p_vega = pos.get("vega", 0.0)
            p_theta = pos.get("theta", 0.0)

            totals["delta"] += p_delta * qty
            totals["gamma"] += p_gamma * qty
            totals["vega"] += p_vega * qty
            totals["theta"] += p_theta * qty

        logger.info("greeks_aggregated", totals=totals)
        return totals

    def run_stress_test(self, spot_move: float, vol_move: float) -> dict[str, float]:
        """
        Estimate P&L impact under a defined stress scenario.
        Using a Taylor Series expansion (Greeks-based).
        """
        greeks = self.aggregate_greeks()

        # dP = Delta * dS + 0.5 * Gamma * dS^2 + Vega * dV + Theta * dt
        # For simplicity, we assume dS is fixed move, dt=0
        delta_pnl = greeks["delta"] * spot_move
        gamma_pnl = 0.5 * greeks["gamma"] * (spot_move**2)
        vega_pnl = greeks["vega"] * vol_move

        total_pnl = delta_pnl + gamma_pnl + vega_pnl

        logger.warning(
            "stress_test_audit", spot_move=spot_move, vol_move=vol_move, total_pnl=total_pnl
        )

        return {
            "total_pnl": total_pnl,
            "delta_impact": delta_pnl,
            "gamma_impact": gamma_pnl,
            "vega_impact": vega_pnl,
        }

class PnLExplainer:
    """
    Production P&L Explain Engine.
    Decomposes realized P&L into Greek-level components for performance attribution.
    """

    def __init__(self, start_greeks: dict[str, float], end_greeks: dict[str, float]):
        self.start = start_greeks
        self.end = end_greeks

    def explain_pnl(self, spot_move: float, vol_move: float, dt: float) -> dict[str, float]:
        """
        P&L Explain (Attribution).
        dP = Delta*dS + 0.5*Gamma*dS^2 + Vega*dV + Theta*dt + Residual
        """
        delta_pnl = self.start.get("delta", 0.0) * spot_move
        gamma_pnl = 0.5 * self.start.get("gamma", 0.0) * (spot_move**2)
        vega_pnl = self.start.get("vega", 0.0) * vol_move
        theta_pnl = self.start.get("theta", 0.0) * dt

        explained_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl

        logger.info(
            "pnl_explained",
            delta=delta_pnl,
            gamma=gamma_pnl,
            vega=vega_pnl,
            theta=theta_pnl,
            total=explained_pnl,
        )

        return {
            "delta_pnl": delta_pnl,
            "gamma_pnl": gamma_pnl,
            "vega_pnl": vega_pnl,
            "theta_pnl": theta_pnl,
            "explained_total": explained_pnl,
        }

    @classmethod
    async def from_service(cls, user_id: str) -> "RiskAttributor":
        """
        Factory method to create a RiskAttributor from real portfolio data.
        """
        from services.api.graphql.resolvers.portfolio_service import service_get_portfolio

        portfolio = await service_get_portfolio(user_id)
        # Flatten strawberry types to dicts for simpler internal handling if needed
        data = []
        if portfolio and portfolio.positions:
            for pos in portfolio.positions:
                data.append(
                    {
                        "symbol": pos.symbol,
                        "quantity": pos.quantity,
                        "delta": pos.delta or 0.0,
                        "gamma": pos.gamma or 0.0,
                        "vega": pos.vega or 0.0,
                        "theta": pos.theta or 0.0,
                        "type": "CALL"
                        if "C" in pos.symbol
                        else ("PUT" if "P" in pos.symbol else "STOCK"),
                    }
                )
        return cls(data)

if __name__ == "__main__":
    
    Production_portfolio = [
        {
            "symbol": "SPX_260320_C_5200",
            "quantity": 500,
            "delta": 0.65,
            "gamma": 0.0012,
            "vega": 450.0,
            "theta": -120.0,
        },
        {
            "symbol": "SPX_260320_P_5000",
            "quantity": -200,
            "delta": -0.15,
            "gamma": 0.0008,
            "vega": 320.0,
            "theta": -85.0,
        },
    ]
    attributor = RiskAttributor(Production_portfolio)
    print(
        "Stress Test Result (Production):",
        attributor.run_stress_test(spot_move=50, vol_move=0.02),
    )
