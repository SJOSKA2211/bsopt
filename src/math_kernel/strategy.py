"""
Dynamic Pricing Engine

Implements dynamic pricing algorithms for SaaS tiers including:
- A/B Testing of pricing strategies
- Customer segmentation based on sensitivity
- Price elasticity tracking
- Automated price adjustments based on usage/competitor data (mocked)
"""

import logging
import random
from enum import Enum
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)


class PricingStrategy(Enum):
    BASE = "base"
    AGGRESSIVE = "aggressive"
    PENETRATION = "penetration"
    PREMIUM = "premium"


class DynamicPricingService:
    """
    Handles optimized SaaS pricing logic.
    """

    def __init__(self) -> None:
        self.experiments: dict[str, dict[str, Any]] = {
            "tier_pricing_v2": {
                "active": True,
                "variants": ["control", "variant_a"],
                "allocations": {"control": 0.5, "variant_a": 0.5},
            }
        }
    def calculate_impact(self, volume: float, avg_daily_volume: float, volatility: float) -> float:
        """
        Square-root market impact model.
        Impact = Y * Volatility * sqrt(Volume / ADV)
        """
        Y = 1.0  # Constant factor (Institutional grade)
        if avg_daily_volume == 0:
            return 0.0
        return Y * volatility * np.sqrt(volume / avg_daily_volume)

    def analyze_elasticity(self, tier: str, volume: float = 1000, adv: float = 100000, vol: float = 0.2) -> float:
        """
        Calculate dynamic price elasticity based on market impact model.
        Returns the expected price change for a given execution volume.
        """
        sensitivity = {"free": 1.5, "pro": 1.0, "enterprise": 0.5}.get(tier, 1.0)
        impact = self.calculate_impact(volume, adv, vol)
        return impact * sensitivity

    def get_user_variant(self, user_id: str, experiment_name: str) -> str:
        """Deterministically assign a user to an A/B test variant."""
        config = self.experiments.get(experiment_name)
        if not config or not config["active"]:
            return "control"

        # Use a local random instance seeded with user_id for determinism without affecting global state
        rng = random.Random(user_id)  # nosec B311
        r = rng.random()  # nosec B311
        cumulative = 0.0
        for variant, allocation in config["allocations"].items():
            cumulative += allocation
            if r <= cumulative:
                return cast(str, variant)
        return "control"

    def calculate_dynamic_price(
        self, base_price: float, user_tier: str, market_demand_factor: float = 1.0
    ) -> float:
        """
        Calculate adjusted price based on demand and tier segmentation.
        """
        # Segment sensitivity: Enterprise users are less sensitive to price changes
        sensitivity = {"free": 1.5, "pro": 1.0, "enterprise": 0.5}.get(user_tier, 1.0)

        # Track elasticity: price_new = price_old * (1 + (demand_factor - 1) / elasticity)
        # Simplified: Adjust based on demand factor and sensitivity
        adjusted_price = base_price * (1 + (market_demand_factor - 1) * (1 / sensitivity))

        return round(adjusted_price, 2)


    def automate_adjustments(self, competitor_prices: list[float]) -> PricingStrategy:
        """Suggest a strategy based on competitor data."""
        avg_comp = np.mean(competitor_prices)
        # Logic to stay competitive
        if avg_comp < 50:
            return PricingStrategy.PENETRATION
        if avg_comp > 200:
            return PricingStrategy.PREMIUM
        return PricingStrategy.BASE


# Global instance
dynamic_pricing = DynamicPricingService()
