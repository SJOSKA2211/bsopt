import re

# Fix "E402 Module level import not at top of file" by adding "# noqa: E402"
with open("src/database/__init__.py", "r") as f:
    c = f.read()
c = c.replace('from collections.abc import Callable', 'from collections.abc import Callable  # noqa: E402')
with open("src/database/__init__.py", "w") as f:
    f.write(c)

with open("src/ingestion/router.py", "r") as f:
    c = f.read()
c = c.replace('from src.shared.observability import (', 'from src.shared.observability import (  # noqa: E402')
with open("src/ingestion/router.py", "w") as f:
    f.write(c)

with open("src/math_kernel/rust_engine.py", "r") as f:
    c = f.read()
c = c.replace('from src.math_kernel.base import PricingStrategy', 'from src.math_kernel.base import PricingStrategy  # noqa: E402')
c = c.replace('from src.math_kernel.models import BSParameters, OptionGreeks', 'from src.math_kernel.models import BSParameters, OptionGreeks  # noqa: E402')
with open("src/math_kernel/rust_engine.py", "w") as f:
    f.write(c)

with open("src/shared/observability.py", "r") as f:
    c = f.read()
c = c.replace('from prometheus_client import Counter, Histogram', 'from prometheus_client import Counter, Histogram  # noqa: E402')
with open("src/shared/observability.py", "w") as f:
    f.write(c)


# Fix the "Expected an indented block after `if` statement" syntax error in anomaly detector
with open("src/ml/aiops/anomaly_detector.py", "r") as f:
    c = f.read()

c = c.replace('        if use_ray and RAY_AVAILABLE and ray.is_initialized(): # type: ignore\n            model_id = ray.put(self.model) # type: ignore', '        if use_ray and RAY_AVAILABLE and ray.is_initialized(): # type: ignore\n            # ray.put(scaled_features)\n            model_id = ray.put(self.model) # type: ignore')
c = c.replace('    class Nn:\n        class Module: pass\n\n    F = None', '    class Nn:\n        class Module:\n            pass\n\n    F = None')

with open("src/ml/aiops/anomaly_detector.py", "w") as f:
    f.write(c)
