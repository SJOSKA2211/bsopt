import os

from src.config import Settings

# Test with a valid generic postgres string
os.environ["DATABASE_URL"] = "postgresql://user:pass@localhost:5432/db"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["JWT_SECRET"] = "secret"

try:
    settings = Settings()
    print("Settings loaded successfully!")
except Exception as e:
    print(f"Failed to load settings: {e}")
    exit(1)
