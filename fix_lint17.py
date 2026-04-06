import re

# Fix email_tasks exception
with open("src/workers/tasks/email_tasks.py", "r") as f:
    c = f.read()
c = c.replace('        except Exception:', '        except Exception as e:')
with open("src/workers/tasks/email_tasks.py", "w") as f:
    f.write(c)

# Fix backtest_evaluator cast
with open("src/ml/aiops/backtest_evaluator.py", "r") as f:
    c = f.read()
c = c.replace('from typing import TypedDict', 'from typing import TypedDict, cast')
with open("src/ml/aiops/backtest_evaluator.py", "w") as f:
    f.write(c)

# Fix main Depends
with open("src/ml/main.py", "r") as f:
    c = f.read()
c = c.replace('from fastapi import FastAPI, Request', 'from fastapi import FastAPI, Request, Depends')
with open("src/ml/main.py", "w") as f:
    f.write(c)

# Remove unused tracking_uri from config.py
with open("src/shared/config.py", "r") as f:
    c = f.read()

# Since line 447 has the duplicate tracking_uri
# I will just remove the whole property block
c = re.sub(r'    @property\n    def tracking_uri\(self\) -> str:\n        """Returns the MLflow tracking URI, defaulting to database backend if not set\."""\n        if self\.MLFLOW_TRACKING_URI:\n            return self\.MLFLOW_TRACKING_URI\n        return f"postgresql\+psycopg2://{self\.DB_USER}:{self\.DB_PASSWORD}@{self\.DB_HOST}:{self\.DB_PORT}/{self\.DB_NAME}"', '', c)
with open("src/shared/config.py", "w") as f:
    f.write(c)
