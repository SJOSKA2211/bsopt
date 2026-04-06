import re

with open("src/ml/aiops/anomaly_detector.py", "r") as f:
    c = f.read()

c = c.replace('            model_id = ray.put(self.model) # type: ignore', '            model_id = ray.put(self.model) # type: ignore\n            _ = model_id')

with open("src/ml/aiops/anomaly_detector.py", "w") as f:
    f.write(c)
