import re

# Fix backtest evaluator
with open("src/ml/aiops/backtest_evaluator.py", "r") as f:
    c = f.read()

c = c.replace('from typing import TypedDict', 'from typing import TypedDict, cast')

with open("src/ml/aiops/backtest_evaluator.py", "w") as f:
    f.write(c)

# Fix email_tasks exception
with open("src/workers/tasks/email_tasks.py", "r") as f:
    c = f.read()

c = re.sub(r'except Exception:', r'except Exception as e:', c)
with open("src/workers/tasks/email_tasks.py", "w") as f:
    f.write(c)

# Remove unused imports in auth/__init__.py
with open("src/auth/__init__.py", "r") as f:
    c = f.read()

c = c.replace('from src.auth.core import hashing as password', 'from src.auth.core import hashing as password\n__all__ = ["password", "mfa"]')

with open("src/auth/__init__.py", "w") as f:
    f.write(c)
