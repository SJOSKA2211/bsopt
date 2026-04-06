with open("src/ml/aiops/backtest_evaluator.py", "r") as f:
    c = f.read()

# Make sure cast is imported
c = c.replace('from typing import TypedDict', 'from typing import TypedDict, cast')

with open("src/ml/aiops/backtest_evaluator.py", "w") as f:
    f.write(c)

with open("src/workers/tasks/email_tasks.py", "r") as f:
    c = f.read()
import re
c = re.sub(r'        except Exception:', '        except Exception as e:', c)
with open("src/workers/tasks/email_tasks.py", "w") as f:
    f.write(c)
