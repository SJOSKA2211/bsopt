import numpy as np
import structlog
from typing import Tuple, Dict, List, Union
from src.ml.drift import calculate_psi, calculate_ks_test
from prometheus_client import Gauge # Assuming Prometheus client is available

logger = structlog.get_logger()

# Define Prometheus gauges for drift detection
PSI_DRIFT_STATUS = Gauge('aiops_psi_drift_status', '1 if PSI drift detected, 0 otherwise', ['feature'])
KS_DRIFT_STATUS = Gauge('aiops_ks_drift_status', '1 if KS drift detected, 0 otherwise', ['feature'])
OVERALL_DRIFT_STATUS = Gauge('aiops_overall_drift_status', '1 if any drift detected, 0 otherwise')

class DataDriftDetector:
    def __init__(self, psi_threshold: float = 0.1, ks_threshold: float = 0.05):
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold

    def detect_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> Tuple[bool, Dict]:
        if reference_data.shape[0] == 0 or current_data.shape[0] == 0:
            raise ValueError("Reference or current data cannot be empty.")
        
        if reference_data.shape[1] != current_data.shape[1]:
            raise ValueError("Reference and current data must have the same number of features.")

        n_features = reference_data.shape[1]
        overall_drift_detected = False
        drift_info: Dict[str, Union[float, bool, List[Dict]]] = {
            "overall_drift_detected": False,
            "feature_drifts": []
        }

        # Handle univariate case (single feature)
        if n_features == 1:
            psi_score = calculate_psi(reference_data.flatten(), current_data.flatten())
            _, ks_p_value = calculate_ks_test(reference_data.flatten(), current_data.flatten())
            
            psi_drift = psi_score >= self.psi_threshold
            ks_drift = ks_p_value <= self.ks_threshold

            if psi_drift or ks_drift:
                overall_drift_detected = True
            
            drift_types = []
            if psi_drift:
                drift_types.append("PSI_Drift")
                PSI_DRIFT_STATUS.labels(feature='overall').set(1)
            else:
                PSI_DRIFT_STATUS.labels(feature='overall').set(0)
            if ks_drift:
                drift_types.append("KS_Drift")
                KS_DRIFT_STATUS.labels(feature='overall').set(1)
            else:
                KS_DRIFT_STATUS.labels(feature='overall').set(0)

            drift_info["PSI"] = psi_score
            drift_info["KS_P_VALUE"] = ks_p_value
            drift_info["drift_detected"] = overall_drift_detected
            drift_info["drift_types"] = drift_types
            
            logger.info("data_drift_univariate_check", psi=psi_score, ks_p=ks_p_value, drift_detected=overall_drift_detected)

        # Handle multivariate case (multiple features)
        else:
            for i in range(n_features):
                feature_ref = reference_data[:, i]
                feature_curr = current_data[:, i]
                
                psi_score = calculate_psi(feature_ref, feature_curr)
                _, ks_p_value = calculate_ks_test(feature_ref, feature_curr)
                
                psi_drift = psi_score >= self.psi_threshold
                ks_drift = ks_p_value <= self.ks_threshold
                
                feature_drift_detected = False
                feature_drift_types = []

                if psi_drift:
                    feature_drift_detected = True
                    feature_drift_types.append("PSI_Drift")
                    PSI_DRIFT_STATUS.labels(feature=f'feature_{i}').set(1)
                else:
                    PSI_DRIFT_STATUS.labels(feature=f'feature_{i}').set(0)
                if ks_drift:
                    feature_drift_detected = True
                    feature_drift_types.append("KS_Drift")
                    KS_DRIFT_STATUS.labels(feature=f'feature_{i}').set(1)
                else:
                    KS_DRIFT_STATUS.labels(feature=f'feature_{i}').set(0)

                if feature_drift_detected:
                    overall_drift_detected = True
                    drift_info["feature_drifts"].append({
                        "feature_index": i,
                        "psi_score": psi_score,
                        "ks_p_value": ks_p_value,
                        "drift_detected": True,
                        "drift_types": feature_drift_types
                    })
                logger.info("data_drift_feature_check", feature_index=i, psi=psi_score, ks_p=ks_p_value, drift_detected=feature_drift_detected)
        
        drift_info["overall_drift_detected"] = overall_drift_detected
        OVERALL_DRIFT_STATUS.set(1 if overall_drift_detected else 0)
        
        return overall_drift_detected, drift_info
