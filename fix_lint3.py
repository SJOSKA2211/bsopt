with open("src/ml/main.py", "r") as f:
    c = f.read()
c = c.replace('from fastapi import FastAPI, Request', 'from fastapi import FastAPI, Request, Depends')
with open("src/ml/main.py", "w") as f:
    f.write(c)

with open("src/shared/config.py", "r") as f:
    c = f.read()
c = c.replace('    @property\n    def tracking_uri(self) -> str:\n        """Returns the MLflow tracking URI, defaulting to database backend if not set."""\n        if self.MLFLOW_TRACKING_URI:\n            return self.MLFLOW_TRACKING_URI\n        return f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"', '')
with open("src/shared/config.py", "w") as f:
    f.write(c)
