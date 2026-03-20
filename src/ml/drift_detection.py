from typing import Any

import pandas as pd
import structlog
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import DataDriftTable, DatasetDriftMetric
from evidently.report import Report

logger = structlog.get_logger(__name__)


class DriftDetector:
    """
    Detects data and target drift in ML features and predictions.
    Implements the 'Continuous Monitoring' pattern from Phase 4.
    """

    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.report = Report(
            metrics=[DataDriftPreset(), TargetDriftPreset(), DataDriftTable(), DatasetDriftMetric()]
        )

    def detect_drift(self, current_data: pd.DataFrame) -> dict[str, Any]:
        """
        Compare current batch of data with reference data to detect drift.
        """
        logger.info("starting_drift_detection", current_rows=len(current_data))

        try:
            self.report.run(reference_data=self.reference_data, current_data=current_data)
            result = self.report.as_dict()

            drift_detected = result["metrics"][3]["result"]["dataset_drift"]
            drift_score = result["metrics"][3]["result"]["drift_share"]

            logger.info(
                "drift_detection_complete", drift_detected=drift_detected, drift_score=drift_score
            )

            return {
                "drift_detected": drift_detected,
                "drift_score": drift_score,
                "metrics": result["metrics"],
            }
        except Exception as e:
            logger.error("drift_detection_failed", error=str(e))
            raise


if __name__ == "__main__":
    import numpy as np

    # 1. Generate reference data (Normal distribution)
    ref_df = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 1000),
            "feature2": np.random.normal(5, 2, 1000),
            "target": np.random.normal(10, 5, 1000),
        }
    )

    # 2. Generate current data with DRIFT (Shifted mean)
    cur_df = pd.DataFrame(
        {
            "feature1": np.random.normal(0.5, 1, 1000),  # Drifted
            "feature2": np.random.normal(5, 2, 1000),  # Stable
            "target": np.random.normal(12, 5, 1000),  # Target Drifted
        }
    )

    detector = DriftDetector(ref_df)
    results = detector.detect_drift(cur_df)

    print(f"Drift Detected: {results['drift_detected']}")
    print(f"Drift Score: {results['drift_score']:.4f}")
