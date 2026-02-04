import numpy as np
import os
import sys
import requests
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ml.trainer import InstrumentedTrainer

def verify_pushgateway(job_name):
    """Verify that metrics exist in Pushgateway."""
    url = "http://localhost:9091/metrics"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            if f'job="{job_name}"' in response.text:
                print(f"SUCCESS: Metrics for job '{job_name}' found in Pushgateway.")
                # Look for specific metrics
                metrics = [
                    "ml_model_accuracy_ratio",
                    "ml_model_rmse_total",
                    "ml_training_duration_seconds"
                ]
                for m in metrics:
                    if m in response.text:
                        print(f" - Metric {m} present.")
                    else:
                        print(f" - WARNING: Metric {m} NOT found.")
                return True
            else:
                print(f"FAILURE: Metrics for job '{job_name}' NOT found in Pushgateway.")
                return False
        else:
            print(f"FAILURE: Pushgateway returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"ERROR: Could not connect to Pushgateway: {e}")
        return False

def verify_prometheus(job_name):
    """Verify that Prometheus has scraped the metrics."""
    # Note: This requires Prometheus to be running and configured to scrape Pushgateway
    url = "http://localhost:9090/api/v1/query"
    params = {'query': f'ml_model_accuracy_ratio{{job="{job_name}"}}'}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success' and len(data['data']['result']) > 0:
                print(f"SUCCESS: Prometheus has successfully scraped metrics for job '{job_name}'.")
                return True
            else:
                print(f"WARNING: Prometheus has NOT YET scraped metrics for job '{job_name}' or query returned no results.")
                return False
        else:
            print(f"FAILURE: Prometheus API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"ERROR: Could not connect to Prometheus: {e}")
        return False

def run_simulation():
    print("Starting ML metrics verification simulation...")
    
    # Generate dummy data
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    study_name = f"verify_metrics_{int(time.time())}"
    trainer = InstrumentedTrainer(study_name=study_name)
    
    params = {
        "framework": "sklearn",
        "n_estimators": 10,
        "max_depth": 3
    }
    
    print(f"Running training for study: {study_name}")
    trainer.train_and_evaluate(X, y, params)
    
    print("\nVerifying Pushgateway...")
    if verify_pushgateway(study_name):
        print("\nVerifying Prometheus (this may take up to 15s for next scrape)...")
        # Retry a few times for Prometheus scrape
        for i in range(5):
            if verify_prometheus(study_name):
                break
            print(f" Retrying in 5s (attempt {i+1}/5)...")
            time.sleep(5)
    else:
        print("Skipping Prometheus check due to Pushgateway failure.")

if __name__ == "__main__":
    # Ensure environment variables for testing
    os.environ["PUSHGATEWAY_URL"] = "localhost:9091"
    run_simulation()
