import re

# Fix math_kernel/implied_vol.py import sorting
with open("src/math_kernel/implied_vol.py", "r") as f:
    c = f.read()
c = c.replace('import numpy as np\nfrom typing import cast\nimport structlog', 'import structlog\nimport numpy as np\nfrom typing import cast')
with open("src/math_kernel/implied_vol.py", "w") as f:
    f.write(c)

# Fix aiops/anomaly_detector.py identation
with open("src/ml/aiops/anomaly_detector.py", "r") as f:
    c = f.read()
c = c.replace('                def remote_detect(chunk, model, engine, threshold):\n                pass', '                def remote_detect(chunk, model, engine, threshold):\n                    pass')
with open("src/ml/aiops/anomaly_detector.py", "w") as f:
    f.write(c)

# Fix backtest_evaluator.py cast
with open("src/ml/aiops/backtest_evaluator.py", "r") as f:
    c = f.read()
if "from typing import TypedDict, cast" not in c:
    c = c.replace("from typing import TypedDict", "from typing import TypedDict, cast")
with open("src/ml/aiops/backtest_evaluator.py", "w") as f:
    f.write(c)

# Fix ml/main.py Depends
with open("src/ml/main.py", "r") as f:
    c = f.read()
if "from fastapi import FastAPI, Request, Depends" not in c:
    c = c.replace("from fastapi import FastAPI, Request", "from fastapi import FastAPI, Request, Depends")
with open("src/ml/main.py", "w") as f:
    f.write(c)

# Fix shared/utils/chaos.py ray
with open("src/shared/utils/chaos.py", "r") as f:
    c = f.read()
c = c.replace('                ray.get_actor(actor_name).exit() # type: ignore', '                if "ray" in globals(): ray.get_actor(actor_name).exit() # type: ignore')
with open("src/shared/utils/chaos.py", "w") as f:
    f.write(c)

# Fix workers/tasks/email_tasks.py
with open("src/workers/tasks/email_tasks.py", "r") as f:
    c = f.read()
c = c.replace('        except Exception:', '        except Exception as e:')
with open("src/workers/tasks/email_tasks.py", "w") as f:
    f.write(c)
