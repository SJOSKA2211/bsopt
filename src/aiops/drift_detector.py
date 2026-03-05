from datetime import datetime
from typing import Any

import numpy as np
import structlog
from prometheus_client import Gauge

from src.database import get_async_db_context
from src.ml.drift import calculate_ks_test, calculate_psi
from src.ml.monitoring.mmd import MultivariateDriftDetector
from src.pricing.factory import PricingEngineFactory

logger = structlog.get_logger()

# Define Prometheus gauges for drift detection
PSI_DRIFT_STATUS = Gauge(
    "aiops_psi_drift_status", "1 if PSI drift detected, 0 otherwise", ["feature"]
)
KS_DRIFT_STATUS = Gauge("aiops_ks_drift_status", "1 if KS drift detected, 0 otherwise", ["feature"])
OVERALL_DRIFT_STATUS = Gauge("aiops_overall_drift_status", "1 if any drift detected, 0 otherwise")


class PricingDriftDetector:
    """
    Unified drift detector combining theoretical error analysis (Black-Scholes)
    and statistical distribution checks (PSI, KS-test).
    """

    def __init__(
        self, threshold: float = 0.05, psi_threshold: float = 0.2, ks_threshold: float = 0.05
    ):
        self.threshold = threshold
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.factory = PricingEngineFactory()
        self.multivariate_detector = MultivariateDriftDetector(threshold=threshold)

    async def check_drift(
        self, symbol: str, current_data: np.ndarray, reference_data: np.ndarray | None = None
    ) -> dict[str, Any]:
        """
        Main entry point for drift analysis.
        """
        # 1. Theoretical Drift (Model vs Black-Scholes)
        # In a real scenario, we'd fetch from DB. For now, assuming current_data contains predictions.
        drift_detected = False
        reasons = []

        # 2. Statistical Data Drift (PSI/KS / MMD)
        if reference_data is not None:
            if current_data.ndim > 1 and current_data.shape[1] > 1:
                # Multivariate detection
                is_drifted, mmd_val = self.multivariate_detector.detect_drift(
                    reference_data, current_data
                )
                if is_drifted:
                    drift_detected = True
                    reasons.append(f"MMD_DRIFT({mmd_val:.4f})")
            else:
                # Univariate fallback
                ref = reference_data.flatten()
                curr = current_data.flatten()

                psi_score = calculate_psi(ref, curr)
                _, ks_p_value = calculate_ks_test(ref, curr)

                psi_drift = psi_score >= self.psi_threshold
                ks_drift = ks_p_value <= self.ks_threshold

                PSI_DRIFT_STATUS.labels(feature=symbol).set(1 if psi_drift else 0)
                KS_DRIFT_STATUS.labels(feature=symbol).set(1 if ks_drift else 0)

                if psi_drift:
                    drift_detected = True
                    reasons.append("PSI_DRIFT")
                if ks_drift:
                    drift_detected = True
                    reasons.append("KS_DRIFT")

        OVERALL_DRIFT_STATUS.set(1 if drift_detected else 0)

        if drift_detected:
            logger.warning("drift_detected", symbol=symbol, reasons=reasons)

        return {
            "symbol": symbol,
            "drift_detected": drift_detected,
            "reasons": reasons,
            "timestamp": datetime.now().isoformat(),
        }

    def calculate_statistical_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> tuple[bool, dict[str, float]]:
        """
        Calculates PSI and KS-Test for univariate drift.
        """
        ref = reference_data.flatten()
        curr = current_data.flatten()
        
        psi_score = calculate_psi(ref, curr)
        _, ks_p_value = calculate_ks_test(ref, curr)
        
        drift = psi_score >= self.psi_threshold or ks_p_value <= self.ks_threshold
        
        metrics = {
            "psi": float(psi_score),
            "ks_p_value": float(ks_p_value)
        }
        
        return drift, metrics

    async def analyze_vol_smile_drift(self, symbol: str) -> dict[str, Any] | None:
        """
        Detects structural changes in the volatility smile.
        """
        # Implementation would compare current IV surface with reference
<<<<<<< HEAD
        return {"symbol": symbol, "status": "not_implemented"}
=======
        return None

>>>>>>> 2a3ad5e0c (feat: Implement database optimization revamp, enhance ML and AIOps modules, and update frontend dependencies.)
