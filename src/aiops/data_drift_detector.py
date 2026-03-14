"""
DataDriftDetector — statistical drift detection for feature distributions.

Supports both univariate and multivariate data using PSI (Population Stability
Index) and the KS (Kolmogorov–Smirnov) two-sample test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import ks_2samp

try:
    import bsopt_core

    _CORE_AVAILABLE = True
except ImportError:
    _CORE_AVAILABLE = False


class DataDriftDetector:
    """
    Detects distribution drift between two datasets using PSI and KS statistics.

    Parameters
    ----------
    psi_threshold : float
        PSI score above which a feature is considered drifted. Defaults to 0.2.
    ks_threshold : float
        KS test p-value *below* which a feature is considered drifted. Defaults to 0.05.
    n_bins : int
        Number of histogram bins used in the PSI calculation inside each feature.
    """

    def __init__(
        self,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,
        n_bins: int = 10,
    ) -> None:
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.n_bins = n_bins

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_drift(
        self, reference: np.ndarray, current: np.ndarray
    ) -> tuple[bool, dict[str, Any]]:
        """
        Compare *reference* and *current* distributions.

        Parameters
        ----------
        reference : np.ndarray, shape (n_samples_ref, n_features) or (n_samples_ref,)
        current   : np.ndarray, shape (n_samples_cur, n_features) or (n_samples_cur,)

        Returns
        -------
        drifted : bool
            True when at least one feature shows significant drift.
        info : dict
            Summary metrics for the caller.  Contains "PSI" (aggregate) and,
            for multivariate input, "feature_drifts" (per-feature detail list).
        """
        ref = np.atleast_2d(np.asarray(reference, dtype=float))
        cur = np.atleast_2d(np.asarray(current, dtype=float))

        # Ensure shapes: (n_samples, n_features)
        if ref.shape[0] == 1 and len(reference) > 1:
            ref = ref.T
        if cur.shape[0] == 1 and len(current) > 1:
            cur = cur.T

        if ref.shape[1] != cur.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: reference has {ref.shape[1]} features, "
                f"current has {cur.shape[1]} features."
            )

        n_features = ref.shape[1]
        feature_drifts: list[dict] = []
        total_psi = 0.0
        any_drifted = False

        for i in range(n_features):
            psi_score = self._psi(ref[:, i], cur[:, i])
            ks_stat, p_val = ks_2samp(ref[:, i], cur[:, i])

            feature_drifted = (psi_score > self.psi_threshold) or (p_val < self.ks_threshold)
            if feature_drifted:
                any_drifted = True

            total_psi += psi_score
            feature_drifts.append(
                {
                    "feature_index": i,
                    "psi": float(psi_score),
                    "ks_stat": float(ks_stat),
                    "ks_p_value": float(p_val),
                    "drifted": feature_drifted,
                }
            )

        info: dict[str, Any] = {
            "PSI": float(total_psi / max(n_features, 1)),
            "feature_drifts": feature_drifts,
            "n_features_drifted": sum(1 for f in feature_drifts if f["drifted"]),
        }

        if n_features == 1:
            # Flatten for the univariate convenience case expected by some tests
            info.update(feature_drifts[0])

        return any_drifted, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Compute the Population Stability Index for a single feature."""
        if _CORE_AVAILABLE:
            try:
                # Use pre-calculated bins for consistency
                bins = np.histogram_bin_edges(reference, bins=self.n_bins).astype(np.float64)
                return float(
                    bsopt_core.calculate_psi(
                        reference.astype(np.float64), current.astype(np.float64), bins
                    )
                )
            except Exception:
                pass

        eps = 1e-8

        # Build bins from the reference distribution
        bins = np.histogram_bin_edges(reference, bins=self.n_bins)

        ref_hist, _ = np.histogram(reference, bins=bins)
        cur_hist, _ = np.histogram(current, bins=bins)

        # Normalize to proportions
        ref_perc = (ref_hist / len(reference)) + eps
        cur_perc = (cur_hist / len(current)) + eps

        psi = float(np.sum((cur_perc - ref_perc) * np.log(cur_perc / ref_perc)))
        return psi
