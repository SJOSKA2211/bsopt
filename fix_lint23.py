with open("src/ml/aiops/backtest_evaluator.py", "r") as f:
    c = f.read()

c = c.replace('from typing import TypedDict, cast', 'from typing import TypedDict, cast')
if 'from typing import cast' not in c and 'cast(' in c:
    c = c.replace('from typing import', 'from typing import cast,')

with open("src/ml/aiops/backtest_evaluator.py", "w") as f:
    f.write(c)

with open("src/workers/tasks/email_tasks.py", "r") as f:
    c = f.read()
c = c.replace('        except Exception:', '        except Exception as e:')
with open("src/workers/tasks/email_tasks.py", "w") as f:
    f.write(c)

with open("src/shared/config.py", "r") as f:
    c = f.read()
import re
c = re.sub(r'    @property\n    def tracking_uri\(self\) -> str:\n        """Returns the MLflow tracking URI, defaulting to database backend if not set\."""\n        if self\.MLFLOW_TRACKING_URI:\n            return self\.MLFLOW_TRACKING_URI\n        return f"postgresql\+psycopg2://{self\.DB_USER}:{self\.DB_PASSWORD}@{self\.DB_HOST}:{self\.DB_PORT}/{self\.DB_NAME}"', '', c)
with open("src/shared/config.py", "w") as f:
    f.write(c)
