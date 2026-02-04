"""
Dynamic Pricing Engine
======================

Implements dynamic pricing algorithms for SaaS tiers including:
- A/B Testing of pricing strategies
- Customer segmentation based on sensitivity
- Price elasticity tracking
- Automated price adjustments based on usage/competitor data (mocked)
"""

import logging
import random
from enum import Enum
from typing import Any, Dict, List, cast

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
        self.experiments: Dict[str, Dict[str, Any]] = {
            "tier_pricing_v2": {
                "active": True,
                "variants": ["control", "variant_a"],
                "allocations": {"control": 0.5, "variant_a": 0.5},
            }
        }
        # Mocked elasticity data: {tier: {price_change: demand_change}}
        self.elasticity_data: Dict[str, List[float]] = {
            "free": [0.0, 0.0],
            "pro": [0.1, -0.05],  # 10% price increase led to 5% demand drop
            "enterprise": [0.2, -0.02],
        }

    def get_user_variant(self, user_id: str, experiment_name: str) -> str:
        """Deterministically assign a user to an A/B test variant."""
        config = self.experiments.get(experiment_name)
        if not config or not config["active"]:
            return "control"

        # Use a local random instance seeded with user_id for determinism without affecting global state
        rng = random.Random(user_id)
        r = rng.random()
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

    def analyze_elasticity(self, tier: str) -> float:
        """Calculate price elasticity of demand: % Change in Q / % Change in P."""
        data = self.elasticity_data.get(tier)
        if not data or data[0] == 0:
            return 0.0
        return data[1] / data[0]

    def automate_adjustments(self, competitor_prices: List[float]) -> PricingStrategy:
        """Suggest a strategy based on competitor data."""
        avg_comp = np.mean(competitor_prices)
        # Logic to stay competitive
        if avg_comp < 50:
            return PricingStrategy.PENETRATION
        elif avg_comp > 200:
            return PricingStrategy.PREMIUM
        return PricingStrategy.BASE


# Global instance
dynamic_pricing = DynamicPricingService()
