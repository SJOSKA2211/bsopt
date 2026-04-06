import re

with open("src/shared/utils/chaos.py", "r") as f:
    c = f.read()
c = c.replace('import asyncio\nimport random\nimport structlog', 'import asyncio\nimport random\nimport structlog\nimport ray')
with open("src/shared/utils/chaos.py", "w") as f:
    f.write(c)

with open("src/ml/aiops/anomaly_detector.py", "r") as f:
    c = f.read()
c = c.replace('use_ray=False', 'use_ray=False')
c = c.replace('        if use_ray and RAY_AVAILABLE and ray.is_initialized():', '        if use_ray:\n            import ray\n            if ray.is_initialized():')
with open("src/ml/aiops/anomaly_detector.py", "w") as f:
    f.write(c)
