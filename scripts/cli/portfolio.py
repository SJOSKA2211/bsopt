"""
CLI Portfolio Manager

Handles local portfolio management for the CLI.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import orjson


@dataclass
class Position:
    """Represents an option position in the portfolio."""

    id: str
    symbol: str
    option_type: str
    quantity: int
    strike: float
    maturity: float
    volatility: float
    rate: float
    dividend: float
    entry_price: float
    entry_date: str
    spot: float


class PortfolioManager:
    """Manages a collection of option positions."""

    def __init__(self) -> None:
        self.portfolio_file = Path.home() / ".bsopt" / "portfolio.json"
        self._ensure_portfolio_dir()
        self.positions: list[Position] = self._load()

    def _ensure_portfolio_dir(self) -> None:
        """Ensure the portfolio directory exists."""
        self.portfolio_file.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[Position]:
        """Load portfolio from file."""
        if not self.portfolio_file.exists():
            return []

        try:
            with open(self.portfolio_file, "rb") as f:
                data = orjson.loads(f.read())
                return [Position(**pos) for pos in data]
        except Exception:
            return []

    def _save(self) -> None:
        """Save current positions to file."""
        with open(self.portfolio_file, "wb") as f:
            f.write(
                orjson.dumps([asdict(pos) for pos in self.positions], option=orjson.OPT_INDENT_2)
            )

    def add_position(self, position: Position) -> None:
        """Add a new position to the portfolio."""
        self.positions.append(position)
        self._save()

    def remove_position(self, position_id: str) -> bool:
        """Remove a position by its ID."""
        initial_count = len(self.positions)
        self.positions = [
            p for p in self.positions if p.id[:8] != position_id and p.id != position_id
        ]
        if len(self.positions) < initial_count:
            self._save()
            return True
        return False

    def list_positions(self) -> list[Position]:
        """Get all positions in the portfolio."""
        return self.positions

    def calculate_position_value(self, position: Position) -> dict[str, Any]:
        """
        Calculate current value and P&L for a position.
        """
        from services.quant.pricing.black_scholes import BlackScholesEngine, BSParameters

        params = BSParameters(
            spot=position.spot,
            strike=position.strike,
            maturity=position.maturity,
            volatility=position.volatility,
            rate=position.rate,
            dividend=position.dividend,
        )

        current_price = float(
            BlackScholesEngine.price(params=params, option_type=position.option_type)
        )

        current_value = current_price * abs(position.quantity) * 100  # Assuming 100 multiplier
        entry_value = position.entry_price * abs(position.quantity) * 100

        pnl = (
            (current_value - entry_value)
            if position.quantity > 0
            else (entry_value - current_value)
        )

        return {
            "current_price": current_price,
            "current_value": current_value,
            "entry_value": entry_value,
            "pnl": pnl,
            "pnl_percent": (pnl / entry_value * 100) if entry_value != 0 else 0,
        }

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Get aggregate metrics for the entire portfolio (OPTIMIZED: Vectorized)."""
        if not self.positions:
            return {
                "pnl": {"total_pnl": 0.0, "total_pnl_percent": 0.0},
                "greeks": {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0},
            }

        from services.quant.pricing.black_scholes import BlackScholesEngine

        # 1. Batch extract params
        spots = np.array([p.spot for p in self.positions], dtype=np.float64)
        strikes = np.array([p.strike for p in self.positions], dtype=np.float64)
        maturities = np.array([p.maturity for p in self.positions], dtype=np.float64)
        vols = np.array([p.volatility for p in self.positions], dtype=np.float64)
        rates = np.array([p.rate for p in self.positions], dtype=np.float64)
        divs = np.array([p.dividend for p in self.positions], dtype=np.float64)
        types = np.array([p.option_type for p in self.positions])
        quantities = np.array([p.quantity for p in self.positions], dtype=np.float64)
        entry_prices = np.array([p.entry_price for p in self.positions], dtype=np.float64)

        # 2. Vectorized Pricing and Greeks
        prices = BlackScholesEngine.price_batch(
            spots, strikes, maturities, vols, rates, divs, types
        )
        greeks = BlackScholesEngine.calculate_greeks(
            spots, strikes, maturities, vols, rates, divs, types
        )

        # 3. Aggregate
        current_values = prices * np.abs(quantities) * 100
        entry_values = entry_prices * np.abs(quantities) * 100

        pnls = np.where(
            quantities > 0, current_values - entry_values, entry_values - current_values
        )

        total_pnl = float(np.sum(pnls))
        total_entry_value = float(np.sum(entry_values))
        total_current_value = float(np.sum(current_values))

        total_delta = float(np.sum(greeks.delta * quantities * 100))
        total_gamma = float(np.sum(greeks.gamma * quantities * 100))
        total_vega = float(np.sum(greeks.vega * quantities * 100))
        total_theta = float(np.sum(greeks.theta * quantities * 100))
        total_rho = float(np.sum(greeks.rho * quantities * 100))

        return {
            "pnl": {
                "total_pnl": total_pnl,
                "total_pnl_percent": (total_pnl / total_entry_value * 100)
                if total_entry_value != 0
                else 0,
                "total_entry_value": total_entry_value,
                "total_current_value": total_current_value,
            },
            "greeks": {
                "delta": total_delta,
                "gamma": total_gamma,
                "vega": total_vega,
                "theta": total_theta,
                "rho": total_rho,
            },
        }
