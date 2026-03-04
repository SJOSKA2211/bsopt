import sys
import os

# The container has /app as PYTHONPATH and src is a package under /app
# So importing from src.ml should work if /app is in sys.path

from src.ml.trainer import ModelTrainer
from src.ml.reinforcement_learning.train import RLTrainer

def test_trainers():
    print("Testing ModelTrainer initialization...")
    try:
        mt = ModelTrainer(study_name="test_ml_study")
        print("ModelTrainer initialized successfully.")
    except Exception as e:
        print(f"ModelTrainer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\nTesting RLTrainer initialization...")
    try:
        rt = RLTrainer(study_name="test_rl_study")
        print("RLTrainer initialized successfully.")
    except Exception as e:
        print(f"RLTrainer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\nAll trainers initialized successfully.")
    return True

if __name__ == "__main__":
    success = test_trainers()
    if not success:
        sys.exit(1)
