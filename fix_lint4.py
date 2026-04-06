with open("src/ml/aiops/anomaly_detector.py", "r") as f:
    c = f.read()

c = c.replace('        if use_ray:\n            import ray\n            if ray.is_initialized():', '        if use_ray and RAY_AVAILABLE:\n            import ray\n            if ray.is_initialized():')
c = c.replace('TORCH_AVAILABLE = False', 'TORCH_AVAILABLE = False\nRAY_AVAILABLE = False')
c = c.replace('            ray.put(scaled_features)', '            # ray.put(scaled_features)')

with open("src/ml/aiops/anomaly_detector.py", "w") as f:
    f.write(c)

with open("src/shared/utils/chaos.py", "r") as f:
    c = f.read()
c = c.replace('import ray', 'import asyncio\nimport random\nimport structlog\ntry:\n    import ray\nexcept ImportError:\n    pass')
with open("src/shared/utils/chaos.py", "w") as f:
    f.write(c)

with open("src/ml/main.py", "r") as f:
    c = f.read()
c = c.replace('from fastapi import FastAPI, Request, Depends', 'from fastapi import FastAPI, Request, Depends')
with open("src/ml/main.py", "w") as f:
    f.write(c)
