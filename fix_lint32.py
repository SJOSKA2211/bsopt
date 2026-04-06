with open("src/ml/aiops/backtest_evaluator.py", "r") as f:
    c = f.read()
if 'from typing import cast' not in c and 'cast(' in c:
    c = c.replace('from typing import TypedDict', 'from typing import TypedDict, cast')
    if 'from typing import TypedDict, cast' not in c:
        c = 'from typing import cast\n' + c
with open("src/ml/aiops/backtest_evaluator.py", "w") as f:
    f.write(c)

with open("src/workers/tasks/email_tasks.py", "r") as f:
    c = f.read()
c = c.replace('        except Exception:', '        except Exception as e:')
with open("src/workers/tasks/email_tasks.py", "w") as f:
    f.write(c)
