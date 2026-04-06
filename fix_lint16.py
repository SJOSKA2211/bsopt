import re

# Fix email tasks exception
with open("src/workers/tasks/email_tasks.py", "r") as f:
    c = f.read()
c = c.replace('        except Exception:', '        except Exception as e:')
with open("src/workers/tasks/email_tasks.py", "w") as f:
    f.write(c)

with open("src/ml/main.py", "r") as f:
    c = f.read()
if "from fastapi import Depends" not in c:
    c = c.replace('from fastapi import FastAPI, Request', 'from fastapi import FastAPI, Request, Depends')
with open("src/ml/main.py", "w") as f:
    f.write(c)

with open("src/ml/aiops/backtest_evaluator.py", "r") as f:
    c = f.read()
c = c.replace('from typing import TypedDict', 'from typing import TypedDict, cast')
with open("src/ml/aiops/backtest_evaluator.py", "w") as f:
    f.write(c)

with open("src/database/__init__.py", "r") as f:
    c = f.read()
c = c.replace('import os\n', '')
with open("src/database/__init__.py", "w") as f:
    f.write(c)
