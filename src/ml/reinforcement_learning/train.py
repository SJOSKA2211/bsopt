import os
import argparse
import numpy as np
import mlflow
import mlflow.pytorch
import tempfile
import structlog
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from src.ml.reinforcement_learning.trading_env import TradingEnvironment
from src.ml.reinforcement_learning.transformer_policy import TransformerSingularityExtractor, TransformerTD3Policy
from src.config import settings
from src.shared.shm_manager import SHMManager
import torch

logger = structlog.get_logger()

class SHMWeightSyncCallback(BaseCallback):
    """
    Synchronizes model weights to shared memory for zero-copy access.
    """
    def __init__(self, shm_name: str = "rl_weights", verbose: int = 0):
        super().__init__(verbose)
        self.shm = SHMManager(shm_name, dict, size=50 * 1024 * 1024)
        self.shm.create()

    def _on_step(self) -> bool:
        if self.num_timesteps % 1000 == 0:
            weights = {k: v.cpu().numpy().tolist() for k, v in self.model.policy.state_dict().items()}
            self.shm.write(weights)
            logger.debug("weights_synced_to_shm", step=self.num_timesteps)
        return True

def train_td3(total_timesteps: int = 10000, model_path: str = "models/best_td3"):
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment("rl_trading_singularity")
    
    with mlflow.start_run() as run:
        try:
            env = TradingEnvironment()
            eval_env = TradingEnvironment()
        except Exception as e:
            logger.error("ml_env_setup_failed", error=str(e))
            raise # Re-raise to crash training if env fails
            
        policy_kwargs = dict(
            features_extractor_class=TransformerSingularityExtractor,
            features_extractor_kwargs=dict(features_dim=512, d_model=256, nhead=8, num_layers=4),
            net_arch=dict(pi=[256, 256], qf=[256, 256])
        )

        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        try:
            model = TD3(
                TransformerTD3Policy, 
                env, 
                action_noise=action_noise, 
                verbose=1,
                policy_kwargs=policy_kwargs,
                learning_rate=1e-4,
                buffer_size=200000,
                batch_size=256,
                tau=0.005,
                gamma=0.99
            )
        except Exception as e:
            logger.error("ml_model_init_failed", error=str(e))
            raise

        eval_callback = EvalCallback(
            eval_env, 
            best_model_save_path=model_path,
            log_path="./logs/results/", 
            eval_freq=max(1, total_timesteps // 10),
            deterministic=True
        )
        
        shm_callback = SHMWeightSyncCallback()
        callback = CallbackList([eval_callback, shm_callback])
        
        logger.info("training_started", total_timesteps=total_timesteps)
        try:
            model.learn(total_timesteps=total_timesteps, callback=callback)
        except Exception as e:
            logger.error("ml_model_learn_failed", error=str(e))
            raise
        
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save(model_path)
            mlflow.pytorch.log_model(model.policy, "model")
        except Exception as e:
            logger.error("ml_model_save_failed", error=str(e))
            raise
        
        return {"run_id": run.info.run_id, "model_path": model_path}

def train_distributed(*args, **kwargs):
    """Alias for train_td3 for compatibility with tests."""
    return train_td3(*args, **kwargs)

def main():
    parser = argparse.ArgumentParser(description="Train RL Trading Singularity")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--output", type=str, default="models/td3_singularity")
    
    args = parser.parse_args()
    train_td3(total_timesteps=args.timesteps, model_path=args.output)

if __name__ == "__main__":
    main()
