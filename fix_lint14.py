with open("src/ml/main.py", "r") as f:
    c = f.read()

# Make sure Depends is imported
if "from fastapi import Depends" not in c and "from fastapi import FastAPI, Request, Depends" not in c:
    c = c.replace("from fastapi import FastAPI, Request", "from fastapi import FastAPI, Request, Depends")
with open("src/ml/main.py", "w") as f:
    f.write(c)

with open("src/shared/utils/chaos.py", "r") as f:
    c = f.read()
c = c.replace('                if "ray" in globals(): ray.get_actor(actor_name).exit() # type: ignore', '                import ray # type: ignore\n                ray.get_actor(actor_name).exit() # type: ignore')
with open("src/shared/utils/chaos.py", "w") as f:
    f.write(c)

with open("src/math_kernel/implied_vol.py", "r") as f:
    c = f.read()
c = c.replace('import structlog\nimport numpy as np\nfrom typing import cast', 'import numpy as np\nimport structlog\nfrom typing import cast')
with open("src/math_kernel/implied_vol.py", "w") as f:
    f.write(c)

with open("src/workers/tasks/email_tasks.py", "r") as f:
    c = f.read()
c = c.replace('        except Exception:', '        except Exception as e:')
with open("src/workers/tasks/email_tasks.py", "w") as f:
    f.write(c)
