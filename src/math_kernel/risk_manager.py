from typing import Any

import structlog

from src.shared.lua_scripts import ADVANCED_RISK_MATRIX

logger = structlog.get_logger(__name__)


class RiskManager:
    """
    Orchestrates Local SHM Risk State and Global Redis LUA State.
    Provides a unified interface for sub-microsecond local checks and atomic distributed sync.
    """

    def __init__(self, redis_client: Any, risk_shm: Any) -> None:
        self.redis = redis_client
        self.risk_shm = risk_shm
        self.kill_switch_key = "risk:kill_switch"
        self.matrix_key = "risk:state:matrix"

    async def global_risk_sync(
        self, d_delta: float, d_gamma: float, d_vega: float, limits: dict[str, float]
    ) -> tuple[bool, str]:
        """
        Atomic global sync via Redis LUA.
        Checks kill-switch and greeks limits.
        """
        try:
            # Keys: [risk_state_hash, global_kill_switch]
            # Args: [d_delta, d_gamma, d_vega, max_d, max_g, max_v]
            result = await self.redis.eval(
                ADVANCED_RISK_MATRIX,
                2,
                self.matrix_key,
                self.kill_switch_key,
                d_delta,
                d_gamma,
                d_vega,
                limits.get("max_delta", 10000.0),
                limits.get("max_gamma", 5000.0),
                limits.get("max_vega", 5000.0),
            )

            if result[0] == 1:
                # SUCCESS: result[1], result[2], result[3] are New Delta, Gamma, Vega
                new_delta, new_gamma, new_vega = (
                    float(result[1]),
                    float(result[2]),
                    float(result[3]),
                )

                # Update local SHM state to prevent drift
                self.risk_shm.update(
                    new_delta,
                    new_gamma,
                    new_vega,
                    limits.get("max_delta", 10000.0),
                    limits.get("max_gamma", 5000.0),
                    limits.get("max_vega", 5000.0),
                )
                # Note: If SHM is extended for gamma/vega, update them here too.

                return True, "SUCCESS"

            reason = result[1].decode() if isinstance(result[1], bytes) else str(result[1])
            return False, reason

        except Exception as e:
            logger.error("global_risk_sync_failed", error=str(e))
            return False, "SYNC_ERROR"

    async def set_kill_switch(self, active: bool) -> None:
        """Emergency trigger to stop all trading."""
        await self.redis.set(self.kill_switch_key, "1" if active else "0")
        logger.warning("risk_kill_switch_updated", active=active)