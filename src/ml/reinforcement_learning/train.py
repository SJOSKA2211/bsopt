import os
import argparse
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
import tempfile
import mlflow
import structlog
from src.ml.reinforcement_learning.trading_env import TradingEnvironment

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logger = structlog.get_logger()

class MLflowMetricsCallback(BaseCallback):
    """
    Custom callback for logging RL metrics to MLflow.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Retrieve metrics from the logger directly
        metrics = self.logger.name_to_value
        if metrics:
            # Filter out metrics that are not suitable for MLflow (e.g., strings)
            mlflow_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, np.float32, np.float64))}
            if mlflow_metrics:
                mlflow.log_metrics(mlflow_metrics, step=self.num_timesteps)
        return True

def train_td3(total_timesteps: int = 100000, model_path: str = "models/td3_trading_agent"):
    """
    Train a TD3 agent on the custom TradingEnvironment using Stable-Baselines3.
    """
    # Create environment
    env = TradingEnvironment()
    eval_env = TradingEnvironment()
    
    # Action noise for TD3 exploration
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    # Initialize MLflow
    mlflow.set_experiment("RL_Trading_Agent")
    
    with mlflow.start_run() as run:
        logger.info("training_started", algorithm="TD3", total_timesteps=total_timesteps)
        
        # Log parameters
        mlflow.log_params({
            "algorithm": "TD3",
            "framework": "stable-baselines3",
            "total_timesteps": total_timesteps,
            "action_noise_sigma": 0.1
        })
        
        # Create model
        model = TD3(
            "MlpPolicy", 
            env, 
            action_noise=action_noise, 
            verbose=1,
            tensorboard_log="./logs/td3_trading/"
        )
        
        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env, 
            best_model_save_path="./models/best_td3/",
            log_path="./logs/results/", 
            eval_freq=max(1, total_timesteps // 10),
            deterministic=True, 
            render=False
        )
        
        # MLflow metrics callback
        mlflow_callback = MLflowMetricsCallback()
        
        # Combine callbacks
        callback = CallbackList([eval_callback, mlflow_callback])
        
        # Train the agent
        model.learn(total_timesteps=total_timesteps, callback=callback)
        
        # Final evaluation result
        final_reward = float(np.mean(eval_callback.last_mean_reward)) if hasattr(eval_callback, 'last_mean_reward') else 0.0
        
        # Save the model
        actual_model_path = model_path
        if not actual_model_path:
            # Generate a unique temp path if none provided
            tmp_dir = tempfile.mkdtemp()
            actual_model_path = os.path.join(tmp_dir, "model.zip")
            
        os.makedirs(os.path.dirname(actual_model_path), exist_ok=True)
        model.save(actual_model_path)
        
        # Log model to MLflow
        mlflow.pytorch.log_model(model.policy, "model")
        
        logger.info("training_completed", run_id=run.info.run_id, model_path=actual_model_path, reward=final_reward)
        
        # Return only picklable metadata
        return {
            "run_id": run.info.run_id,
            "model_path": actual_model_path,
            "mean_reward": final_reward
        }

@ray.remote
def train_td3_remote(total_timesteps: int = 10000, model_path: str = None):
    """Ray remote function for distributed training."""
    # Ensure each remote task has its own temp model path if not specified
    return train_td3(total_timesteps=total_timesteps, model_path=model_path)

def train_distributed(num_instances: int = 2, total_timesteps: int = 10000, ray_address: str = "auto"):
    """
    Run multiple training instances in parallel using Ray and select the best one.
    """
    if not RAY_AVAILABLE:
        logger.error("ray_not_available", message="Ray not installed. Cannot run distributed training.")
        return None

    # Connect to Ray cluster
    ray.init(address=ray_address, ignore_reinit_error=True)
    
    logger.info("distributed_training_started", num_instances=num_instances, total_timesteps=total_timesteps)
    
    # Launch parallel training tasks
    futures = [train_td3_remote.remote(total_timesteps=total_timesteps) for _ in range(num_instances)]
    results = ray.get(futures)
    
    # Find the best result based on mean_reward
    best_result = max(results, key=lambda x: x["mean_reward"])
    
    logger.info("distributed_training_completed", 
                num_instances=num_instances, 
                best_reward=best_result["mean_reward"],
                best_model=best_result["model_path"])
                
    return results, best_result

def main():
    parser = argparse.ArgumentParser(description="Train RL Trading Agent")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total training timesteps")
    parser.add_argument("--output", type=str, default="models/td3_trading_agent", help="Path to save the model")
    parser.add_argument("--distributed", action="store_true", help="Use Ray for parallel training instances")
    parser.add_argument("--instances", type=int, default=2, help="Number of parallel instances")
    parser.add_argument("--ray-address", type=str, default="auto", help="Ray cluster address")
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    if args.distributed:
        train_distributed(num_instances=args.instances, total_timesteps=args.timesteps, ray_address=args.ray_address)
    else:
        train_td3(total_timesteps=args.timesteps, model_path=args.output)

if __name__ == "__main__":
    main()