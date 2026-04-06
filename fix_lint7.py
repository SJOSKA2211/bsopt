import re

with open("src/ml/aiops/anomaly_detector.py", "r") as f:
    c = f.read()

c = c.replace('        if use_ray and RAY_AVAILABLE and ray.is_initialized(): # type: ignore\n                logger.info("training_anomaly_detector_with_ray", engine=self.engine)', '        if use_ray and RAY_AVAILABLE and ray.is_initialized(): # type: ignore\n            logger.info("training_anomaly_detector_with_ray", engine=self.engine)')
c = c.replace('        if use_ray and RAY_AVAILABLE and ray.is_initialized(): # type: ignore\n                # Ray-based distributed inference', '        if use_ray and RAY_AVAILABLE and ray.is_initialized(): # type: ignore\n            # Ray-based distributed inference')

with open("src/ml/aiops/anomaly_detector.py", "w") as f:
    f.write(c)
