"""
PricingDriftDetector — Entry point for distribution-based drift detection.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

import numpy as np
import structlog

from src.ml.aiops.data_drift_detector import DataDriftDetector

logger = structlog.get_logger(__name__)

class PricingDriftDetector:
    """
    Specialized drift detector for pricing models.
    """

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold
        self.logger = logger
        self.detector = DataDriftDetector(psi_threshold=threshold)

    async def check_drift(
        self, symbol: str, current_data: np.ndarray, reference_data: np.ndarray
    ) -> dict[str, Any]:
        """
        Check for drift in pricing data.
        """
        # Ensure data is numpy array
        ref = np.asarray(reference_data)
        cur = np.asarray(current_data)
        
        drifted, info = self.detector.detect_drift(ref, cur)
        
        reasons = []
        if drifted:
            reasons.append(f"PSI score {info.get('PSI', 0.0):.4f} exceeded threshold {self.threshold}")
            
        return {
            "symbol": symbol,
            "drift_detected": drifted,
            "reasons": reasons,
            "info": info
        }

async def run_cli():
    parser = argparse.ArgumentParser(description="Pricing Drift Detector")
    parser.add_argument("--ticker", required=True, help="Ticker symbol")
    parser.add_argument("--tracking_uri", default="http://mlflow:5000", help="MLflow tracking URI")
    args = parser.parse_args()

    # Mock data for demonstration if not provided
    # In a real scenario, this would fetch data from MLflow or Feature Store
    reference_data = np.random.normal(0, 1, (100, 5))
    current_data = np.random.normal(0.1, 1.1, (100, 5))

    detector = PricingDriftDetector()
    result = await detector.check_drift(args.ticker, current_data, reference_data)
    
    if result["drift_detected"]:
        logger.warning("drift_detected", **result)
    else:
        logger.info("no_drift_detected", **result)

if __name__ == "__main__":
    asyncio.run(run_cli())
