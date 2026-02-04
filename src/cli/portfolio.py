"""
CLI Portfolio Manager
=====================

Handles local portfolio management for the CLI.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, cast


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
        self.positions: List[Position] = self._load()

    def _ensure_portfolio_dir(self) -> None:
        """Ensure the portfolio directory exists."""
        self.portfolio_file.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> List[Position]:
        """Load portfolio from file."""
        if not self.portfolio_file.exists():
            return []

        try:
            with open(self.portfolio_file, "r") as f:
                data = json.load(f)
                return [Position(**pos) for pos in data]
        except Exception:
            return []

    def _save(self) -> None:
        """Save current positions to file."""
        with open(self.portfolio_file, "w") as f:
            json.dump([asdict(pos) for pos in self.positions], f, indent=2)

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

    def list_positions(self) -> List[Position]:
        """Get all positions in the portfolio."""
        return self.positions

    def calculate_position_value(self, position: Position) -> Dict[str, Any]:
        """
        Calculate current value and P&L for a position.

        This is a mock implementation. In reality, it would use the pricing engines.
        """
        from src.pricing.black_scholes import BlackScholesEngine, BSParameters

        params = BSParameters(
            spot=position.spot,
            strike=position.strike,
            maturity=position.maturity,
            volatility=position.volatility,
            rate=position.rate,
            dividend=position.dividend,
        )

        if position.option_type == "call":
            current_price = float(BlackScholesEngine.price(params=params, option_type="call"))
        else:
            current_price = float(BlackScholesEngine.price(params=params, option_type="put"))

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

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get aggregate metrics for the entire portfolio."""
        total_pnl = 0.0
        total_entry_value = 0.0
        total_current_value = 0.0
        total_delta = 0.0

        for pos in self.positions:
            val = self.calculate_position_value(pos)
            total_pnl += val["pnl"]
            total_entry_value += val["entry_value"]
            total_current_value += val["current_value"]

            # Simple delta sum (ignoring contract multiplier for simplicity in mock)
            from src.pricing.black_scholes import BlackScholesEngine, BSParameters, OptionGreeks

            params = BSParameters(
                spot=pos.spot,
                strike=pos.strike,
                maturity=pos.maturity,
                volatility=pos.volatility,
                rate=pos.rate,
                dividend=pos.dividend,
            )
            greeks_res = BlackScholesEngine.calculate_greeks(params, pos.option_type)
            greeks = cast(OptionGreeks, greeks_res)
            total_delta += float(greeks.delta) * pos.quantity * 100

        return {
            "pnl": {
                "total_pnl": total_pnl,
                "total_pnl_percent": (
                    (total_pnl / total_entry_value * 100) if total_entry_value != 0 else 0
                ),
                "total_entry_value": total_entry_value,
                "total_current_value": total_current_value,
            },
            "greeks": {
                "delta": total_delta,
                "gamma": 0.0,  # Mock
                "vega": 0.0,  # Mock
                "theta": 0.0,  # Mock
                "rho": 0.0,  # Mock
            },
        }
