from .anomaly_detector import AnomalyDetector


class TransformerAnomalyDetector(AnomalyDetector):
    """Shim for TransformerAnomalyDetector using the unified AnomalyDetector."""
    def __init__(self, **kwargs):
        super().__init__(engine="transformer", **kwargs)
    
    def train_on_data(self, data, epochs=20):
        return self.train(data, epochs=epochs)