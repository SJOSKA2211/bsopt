with open("src/workers/tasks/email_tasks.py", "r") as f:
    c = f.read()
import re
c = re.sub(r'        except BaseException as e:', '        except Exception as e:', c)
with open("src/workers/tasks/email_tasks.py", "w") as f:
    f.write(c)

with open("src/ml/aiops/backtest_evaluator.py", "r") as f:
    c = f.read()

c = c.replace('from typing import cast\n"""\nAutomated Model Backtesting & Rollback Evaluator', '"""\nAutomated Model Backtesting & Rollback Evaluator')
c = c.replace('import mlflow', 'from typing import cast\nimport mlflow')
with open("src/ml/aiops/backtest_evaluator.py", "w") as f:
    f.write(c)
