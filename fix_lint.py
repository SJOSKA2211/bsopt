import re

# src/ml/main.py
with open("src/ml/main.py", "r") as f:
    c = f.read()
c = c.replace('from fastapi import FastAPI, Request', 'from fastapi import FastAPI, Request, Depends')
with open("src/ml/main.py", "w") as f:
    f.write(c)

# src/portfolio/engine.py
with open("src/portfolio/engine.py", "r") as f:
    c = f.read()
c = c.replace('import pandas as pd', 'import pandas as pd\nfrom typing import Any')
with open("src/portfolio/engine.py", "w") as f:
    f.write(c)

# src/ml/aiops/backtest_evaluator.py
with open("src/ml/aiops/backtest_evaluator.py", "r") as f:
    c = f.read()
c = c.replace('from typing import TypedDict', 'from typing import TypedDict, cast')
with open("src/ml/aiops/backtest_evaluator.py", "w") as f:
    f.write(c)

# src/shared/schemas/market.py
with open("src/shared/schemas/market.py", "r") as f:
    c = f.read()
c = c.replace('percentChange: str', 'percent_change: str')
with open("src/shared/schemas/market.py", "w") as f:
    f.write(c)

# src/workers/tasks/email_tasks.py
with open("src/workers/tasks/email_tasks.py", "r") as f:
    c = f.read()
c = c.replace('        except Exception:', '        except Exception as e:')
with open("src/workers/tasks/email_tasks.py", "w") as f:
    f.write(c)

# src/ml/reinforcement_learning/distributed_trainer.py
with open("src/ml/reinforcement_learning/distributed_trainer.py", "r") as f:
    c = f.read()
c = c.replace('batch_samples // 64', 'self.model.batch_size // 64')
with open("src/ml/reinforcement_learning/distributed_trainer.py", "w") as f:
    f.write(c)

# src/shared/config.py
with open("src/shared/config.py", "r") as f:
    c = f.read()
c = re.sub(r'    @property\n    def tracking_uri\(self\) -> str:\n        """Returns the MLflow tracking URI, defaulting to database backend if not set."""\n        if self\.MLFLOW_TRACKING_URI:\n            return self\.MLFLOW_TRACKING_URI\n        return f"postgresql\+psycopg2://{self\.DB_USER}:{self\.DB_PASSWORD}@{self\.DB_HOST}:{self\.DB_PORT}/{self\.DB_NAME}"', '', c)
with open("src/shared/config.py", "w") as f:
    f.write(c)
