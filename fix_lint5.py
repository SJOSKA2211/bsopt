with open("src/ml/aiops/anomaly_detector.py", "r") as f:
    c = f.read()

c = c.replace('        if use_ray and RAY_AVAILABLE:\n            import ray\n            if ray.is_initialized():\n            logger.info("training_anomaly_detector_with_ray", engine=self.engine)', '        if use_ray and RAY_AVAILABLE:\n            import ray\n            if ray.is_initialized():\n                logger.info("training_anomaly_detector_with_ray", engine=self.engine)')

c = c.replace('        if use_ray and RAY_AVAILABLE:\n            import ray\n            if ray.is_initialized():\n            # Ray-based distributed inference', '        if use_ray and RAY_AVAILABLE:\n            import ray\n            if ray.is_initialized():\n                # Ray-based distributed inference')

with open("src/ml/aiops/anomaly_detector.py", "w") as f:
    f.write(c)
