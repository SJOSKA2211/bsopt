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
        # Bolt Performance Optimization: Accumulate values in local variables and separate
        # conditional logic branches to minimize dictionary lookups and membership checks
        # inside the high-frequency loop. Reduces execution time significantly.
        delta_total = 0.0
        gamma_total = 0.0
        vega_total = 0.0
        theta_total = 0.0

        for pos in self.portfolio:
            qty = pos.get("quantity", 0)
            if not qty:
                continue

            pos_type = pos.get("type", "")

            # Options have specific Greeks; linear assets (stock/crypto) have Delta=1.0
            if pos_type == "CALL" or pos_type == "PUT":
                delta_total += pos.get("delta", 0.0) * qty
                gamma_total += pos.get("gamma", 0.0) * qty
                vega_total += pos.get("vega", 0.0) * qty
                theta_total += pos.get("theta", 0.0) * qty
            else:
                delta_total += pos.get("delta", 1.0) * qty
                gamma_total += pos.get("gamma", 0.0) * qty
                vega_total += pos.get("vega", 0.0) * qty
                theta_total += pos.get("theta", 0.0) * qty

        totals = {
            "delta": delta_total,
            "gamma": gamma_total,
            "vega": vega_total,
            "theta": theta_total,
        }
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
        from api.graphql.resolvers.portfolio_service import service_get_portfolio

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
