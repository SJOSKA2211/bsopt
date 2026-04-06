with open("src/ml/aiops/backtest_evaluator.py", "r") as f:
    c = f.read()
if 'from typing import TypedDict' in c and 'cast' not in c.split('from typing import TypedDict')[1].split('\n')[0]:
    c = c.replace('from typing import TypedDict', 'from typing import TypedDict, cast')
with open("src/ml/aiops/backtest_evaluator.py", "w") as f:
    f.write(c)

with open("src/workers/tasks/email_tasks.py", "r") as f:
    c = f.read()
c = c.replace('except Exception:', 'except Exception as e:')
with open("src/workers/tasks/email_tasks.py", "w") as f:
    f.write(c)
