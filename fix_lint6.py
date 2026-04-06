import re

with open("src/ml/aiops/anomaly_detector.py", "r") as f:
    c = f.read()
c = c.replace('RAY_AVAILABLE = False', 'RAY_AVAILABLE = False\ntry:\n    import ray\n    RAY_AVAILABLE = True\nexcept ImportError:\n    pass')
c = c.replace('        if use_ray and RAY_AVAILABLE and ray.is_initialized():', '        if use_ray and RAY_AVAILABLE and ray.is_initialized(): # type: ignore')
c = c.replace('            data_id = ray.put(scaled_features)', '            data_id = ray.put(scaled_features) # type: ignore\n            _ = data_id')
c = c.replace('            model_id = ray.put(self.model)', '            model_id = ray.put(self.model) # type: ignore')
c = c.replace('            @ray.remote', '            @ray.remote # type: ignore')
c = c.replace('            futures = [remote_detect.remote(c, model_id, self.engine, getattr(self, "threshold", None)) for c in chunks]', '            futures = [remote_detect.remote(c, model_id, self.engine, getattr(self, "threshold", None)) for c in chunks]\n            _ = futures')
c = c.replace('    class nn:\n        class Module: pass\n    F = None', '    class nn:\n        class Module: pass\n\n    F = None')
with open("src/ml/aiops/anomaly_detector.py", "w") as f:
    f.write(c)


with open("src/shared/utils/chaos.py", "r") as f:
    c = f.read()
c = c.replace('                ray.get_actor(actor_name).exit()', '                ray.get_actor(actor_name).exit() # type: ignore')
with open("src/shared/utils/chaos.py", "w") as f:
    f.write(c)

with open("src/ml/main.py", "r") as f:
    c = f.read()
c = c.replace('from fastapi import FastAPI, Request', 'from fastapi import FastAPI, Request, Depends')
with open("src/ml/main.py", "w") as f:
    f.write(c)
