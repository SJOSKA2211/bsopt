#!/usr/bin/env python3
"""
MLflow Watchdog & Auto-Recovery Service
===================================================
Continuously polls MLflow. If a Ray training instance fails, it logs the event,
adapts the hyperparameters (e.g., reduces batch size, adjusts learning rate),
and automatically respawns the training job via Ray.
"""

import os
import subprocess
import time

import mlflow
import ray
import structlog

logger = structlog.get_logger(__name__)

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "bsopt_training")
POLL_INTERVAL_SEC = 60

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def init_ray():
    """Initializes connection to the Ray cluster if not already connected."""
    if not ray.is_initialized():
        ray_address = os.getenv("RAY_ADDRESS", "auto")
        try:
            ray.init(address=ray_address, ignore_reinit_error=True)
            logger.info("ray_cluster_connected", address=ray_address)
        except Exception as e:
            logger.error("ray_cluster_connection_failed", error=str(e))

def get_experiment_id() -> str | None:
    """Gets the experiment ID by name."""
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp:
        return exp.experiment_id
    logger.warning("experiment_not_found", experiment_name=EXPERIMENT_NAME)
    return None

def adapt_parameters(failed_run_params: dict) -> dict:
    """
    Heuristic-based parameter adaptation to recover from failure.
    Reduces batch size to prevent OOM, adjusts learning rate.
    """
    adapted = failed_run_params.copy()
    
    # Adapt Batch Size (Handle Out of Memory issues)
    if "batch_size" in adapted:
        try:
            current_bs = int(adapted["batch_size"])
            new_bs = max(16, current_bs // 2)
            adapted["batch_size"] = str(new_bs)
            logger.info("adapted_batch_size", old=current_bs, new=new_bs)
        except ValueError:
            pass
            
    # Adapt Learning Rate (Handle exploding gradients)
    if "learning_rate" in adapted:
        try:
            current_lr = float(adapted["learning_rate"])
            new_lr = current_lr * 0.5
            adapted["learning_rate"] = str(new_lr)
            logger.info("adapted_learning_rate", old=current_lr, new=new_lr)
        except ValueError:
            pass

    return adapted

def respawn_training_job(params: dict, run_name: str):
    """
    Respawns the Ray training job with adapted parameters.
    """
    logger.info("respawning_training_job", run_name=run_name, params=params)
    
    # Build command line arguments from params
    cmd = ["python", "-m", "src.ml.training.train_all"]
    for k, v in params.items():
        cmd.extend([f"--{k}", str(v)])
        
    try:
        # We launch the training script asynchronously
        subprocess.Popen(cmd, env=os.environ.copy())
        logger.info("training_job_respawned_successfully", cmd=" ".join(cmd))
    except Exception as e:
        logger.error("failed_to_respawn_training_job", error=str(e))

def run_watchdog():
    """Main polling loop."""
    logger.info("mlflow_watchdog_started", tracking_uri=MLFLOW_TRACKING_URI)
    init_ray()
    
    # Keep track of handled failed runs to avoid infinite respawn loops
    handled_runs = set()
    
    while True:
        try:
            exp_id = get_experiment_id()
            if not exp_id:
                time.sleep(POLL_INTERVAL_SEC)
                continue
                
            # Query for failed runs in the last 24 hours
            runs = mlflow.search_runs(
                experiment_ids=[exp_id],
                filter_string="status = 'FAILED'",
                order_by=["start_time DESC"],
                max_results=50
            )
            
            for index, run in runs.iterrows():
                run_id = run["run_id"]
                run_name = run.get("tags.mlflow.runName", f"run_{run_id}")
                
                if run_id in handled_runs:
                    continue
                    
                logger.warning("failed_run_detected", run_id=run_id, run_name=run_name)
                
                # Extract parameters from the run (columns starting with 'params.')
                params = {
                    col.replace("params.", ""): run[col]
                    for col in run.index if col.startswith("params.") and not isinstance(run[col], float) and run[col] is not None
                }
                
                adapted_params = adapt_parameters(params)
                
                # Tag the old run as 'handled_by_watchdog'
                mlflow.tracking.MlflowClient().set_tag(run_id, "watchdog_handled", "true")
                
                # Respawn
                respawn_training_job(adapted_params, run_name=f"{run_name}_retry")
                handled_runs.add(run_id)

        except Exception as e:
            logger.error("watchdog_polling_error", error=str(e))
            
        time.sleep(POLL_INTERVAL_SEC)

if __name__ == "__main__":
    run_watchdog()
