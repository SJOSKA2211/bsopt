"""
DataDriftDetector — institutional wrapper for statistical drift detection.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.ml.drift import calculate_ks_test, calculate_psi

class DataDriftDetector:
    """
    Institutional wrapper for distribution drift detection using PSI and KS.
    """

    def __init__(
        self,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,
    ) -> None:
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold

    def detect_drift(
        self, reference: np.ndarray, current: np.ndarray
    ) -> tuple[bool, dict[str, Any]]:
        """
        Compare reference and current distributions.
        """
        ref = np.atleast_2d(np.asarray(reference, dtype=float))
        cur = np.atleast_2d(np.asarray(current, dtype=float))

        if ref.shape[1] != cur.shape[1]:
            raise ValueError(f"Feature dimension mismatch: {ref.shape[1]} vs {cur.shape[1]}")

        n_features = ref.shape[1]
        feature_drifts: list[dict[str, Any]] = []
        any_drifted = False

        for i in range(n_features):
            psi_score = calculate_psi(ref[:, i], cur[:, i])
            _, p_val = calculate_ks_test(ref[:, i], cur[:, i])

            feature_drifted = (psi_score > self.psi_threshold) or (p_val < self.ks_threshold)
            if feature_drifted:
                any_drifted = True

            feature_drifts.append(
                {
                    "feature_index": i,
                    "psi": float(psi_score),
                    "ks_p_value": float(p_val),
                    "drifted": feature_drifted,
                }
            )

        info = {
            "PSI": float(np.mean([f["psi"] for f in feature_drifts])),
            "feature_drifts": feature_drifts,
            "n_features_drifted": sum(1 for f in feature_drifts if f["drifted"]),
        }

        return any_drifted, info
