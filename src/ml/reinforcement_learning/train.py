import os
import argparse
import numpy as np
import mlflow
import mlflow.pytorch
import ray
import tempfile
import structlog
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from src.ml.reinforcement_learning.trading_env import TradingEnvironment
from src.ml.reinforcement_learning.transformer_policy import TransformerFeatureExtractor
from src.config import get_settings

import struct
from multiprocessing import shared_memory
import io
import torch

logger = structlog.get_logger()

class MLflowMetricsCallback(BaseCallback):
    """
    Callback for logging metrics to MLflow.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log reward and other metrics
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    episode = info["episode"]
                    mlflow.log_metric("episode_reward", episode["r"], step=self.num_timesteps)
                    mlflow.log_metric("episode_length", episode["l"], step=self.num_timesteps)
        return True

class SyncToSHMCallback(BaseCallback):
    """🚀 SINGULARITY: Zero-copy model weight synchronization via Shared Memory."""
    def __init__(self, shm_name: str = "rl_model_weights", verbose: int = 0):
        super().__init__(verbose)
        self.shm_name = shm_name
        self._shm = None

    def _on_step(self) -> bool:
        return True

    def sync_weights(self):
        """🚀 SOTA: Push model state to SHM."""
        # Serialize state_dict to bytes
        buffer = io.BytesIO()
        torch.save(self.model.policy.state_dict(), buffer)
        data = buffer.getvalue()
        
        # Ensure SHM is initialized
        if self._shm is None:
            try:
                self._shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=len(data) + 8)
            except FileExistsError:
                self._shm = shared_memory.SharedMemory(name=self.shm_name)
        
        # Write size and data
        self._shm.buf[:8] = struct.pack("q", len(data))
        self._shm.buf[8:8+len(data)] = data
        logger.info("model_weights_synced_to_shm", size=len(data))

def train_td3(total_timesteps: int = 10000, model_path: str = "models/best_td3"):
    """
    Train TD3 agent with Transformer-based temporal feature extraction.
    """
    settings = get_settings()
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment("rl_trading_agent_transformer")
    
    with mlflow.start_run() as run:
        env = TradingEnvironment()
        eval_env = TradingEnvironment() # separate env for evaluation
        
        # 🚀 SOTA: Policy configuration with Transformer extractor
        policy_kwargs = dict(
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[128, 128], qf=[128, 128])
        )

        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        model = TD3(
            "MlpPolicy", 
            env, 
            action_noise=action_noise, 
            verbose=1,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=128,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1
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
        logger.info("training_started", total_timesteps=total_timesteps)
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

if __name__ == "__main__":
    main()