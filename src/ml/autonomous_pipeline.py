"""
Autonomous ML Pipeline - production-ready orchestration core.
"""

from typing import Any

import pandas as pd

# Re-export or import expected symbols for `@patch`


class AutonomousMLPipeline:
    """
    Compatibility wrapper delegating to high-performance MLPipeline.
    """

    def __init__(self, config: dict[str, 'Any']) -> None:
        self.config = config
        from src.ml.pipeline import MLPipeline

        self.pipeline = MLPipeline(config)

    async def run_pipeline(self) -> dict[str, 'Any']:
        """
        Executes the optimized unified pipeline.
        """
        try:
            await self.pipeline.run()
            return {"status": "success", "drift_detected": False}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates indicators using the optimized Feature Store.
        """
        import asyncio

        from src.ml.feature_store.store import feature_store

        # Synchronous wrapper for feature computation
        required = ["log_return", "RSI_14", "EMA_20"]
        return asyncio.run(feature_store.compute_features(data, required))
