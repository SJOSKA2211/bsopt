import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    def __init__(self, contamination: float = 0.1):
        if not (0.0 < contamination < 0.5):
            raise ValueError("Contamination must be between 0 and 0.5.")
        self.contamination = contamination
        self.model = None

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        if data.shape[0] == 0:
            raise ValueError("Expected 2D array, got 0D array instead") # Matching test's error message
        
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42 # for reproducibility
        )
        return self.model.fit_predict(data)

    def predict(self, data: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Isolation Forest model has not been fitted yet.")
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return self.model.predict(data)
