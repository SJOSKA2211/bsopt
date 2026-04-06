import re

# Fix email_tasks exception correctly this time
with open("src/workers/tasks/email_tasks.py", "r") as f:
    c = f.read()

c = re.sub(r'except Exception:', r'except Exception as e:', c)
with open("src/workers/tasks/email_tasks.py", "w") as f:
    f.write(c)

with open("src/ml/aiops/backtest_evaluator.py", "r") as f:
    c = f.read()

if 'from typing import cast' not in c and 'from typing import TypedDict, cast' not in c:
    c = c.replace('from typing import TypedDict', 'from typing import TypedDict, cast')

with open("src/ml/aiops/backtest_evaluator.py", "w") as f:
    f.write(c)
