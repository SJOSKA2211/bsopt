import argparse
import multiprocessing
import os
from typing import Any

import mlflow
import mlflow.pytorch
import numpy as np
import structlog
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise

from src.shared.config import settings
from src.ml.reinforcement_learning.trading_env import TradingEnvironment
from src.ml.reinforcement_learning.transformer_policy import (
    TransformerFeatureExtractor,
    TransformerTD3Policy,
)
from src.ml.training.base import BaseTrainer
from src.shared.shm_manager import SHMManager

logger = structlog.get_logger()

# Global lock for SHM writes
shm_lock = multiprocessing.Lock()


class SHMWeightSyncCallback(BaseCallback):
    """
    Synchronizes model weights to shared memory.
    OPTIMIZED: Reduced allocation overhead by reusing buffers if possible.
    Includes multiprocessing lock for safe distributed writes.
    """

    def __init__(self, shm_name: str = "rl_weights", sync_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.shm = SHMManager(shm_name, dict, size=50 * 1024 * 1024)
        self.shm.create()
        self.sync_freq = sync_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.sync_freq == 0:
            with torch.no_grad():
                state = {
                    k: v.detach().cpu().numpy() for k, v in self.model.policy.state_dict().items()
                }
            with shm_lock:
                self.shm.write(state)
            logger.info("weights_synced_to_shm", step=self.num_timesteps)
        return True


class RLTrainer(BaseTrainer):
    """
    Unified RL Trainer for BS-OPT.
    """

    def __init__(self, study_name: str, tracking_uri: str | None = None):
        super().__init__(study_name, tracking_uri)
        from src.ml.tracker import ExperimentTracker

        self.tracker = ExperimentTracker(study_name, self.tracking_uri)

    def train_and_evaluate(
        self,
        total_timesteps: int = 10000,
        model_path: str = "models/best_td3",
        warm_start_path: str | None = None,
        sync_freq: int = 1000,
    ) -> dict[str, Any]:
        """
        Executes RL training using TD3 with Transformer policy.
        """
        with self.tracker.start_run(nested=True) as run:
            try:
                env = TradingEnvironment()
                eval_env = TradingEnvironment()
            except Exception as e:
                logger.error("env_setup_failed", error=str(e))
                raise

            policy_kwargs = dict(
                features_extractor_class=TransformerFeatureExtractor,
                features_extractor_kwargs=dict(
                    features_dim=512, d_model=256, nhead=8, num_layers=4
                ),
                net_arch=dict(pi=[256, 256], qf=[256, 256]),
            )

            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )

            try:
                if warm_start_path and os.path.exists(warm_start_path):
                    logger.info("warm_start_active", path=warm_start_path)
                    model = TD3.load(
                        warm_start_path,
                        env=env,
                        policy_kwargs=policy_kwargs,
                        action_noise=action_noise,
                    )
                else:
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
                        gamma=0.99,
                    )
            except Exception as e:
                logger.error("model_init_failed", error=str(e))
                raise

            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=model_path,
                log_path="./logs/results/",
                eval_freq=max(1, total_timesteps // 10),
                deterministic=True,
            )

            shm_callback = SHMWeightSyncCallback(sync_freq=sync_freq)
            callback = CallbackList([eval_callback, shm_callback])

            logger.info("training_active", steps=total_timesteps)
            try:
                model.learn(total_timesteps=total_timesteps, callback=callback)
            except Exception as e:
                logger.error("training_error", error=str(e))
                raise

            try:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                model.save(model_path)

                #  HIGH-PERFORMANCE: High-fidelity model logging
                mlflow.log_params(
                    {
                        "timesteps": total_timesteps,
                        "batch_size": 256,
                        "learning_rate": 1e-4,
                        "policy": "TransformerTD3Policy",
                        "features_dim": 512,
                    }
                )

                # Log final metrics if available
                if hasattr(eval_callback, "last_mean_reward"):
                    mlflow.log_metric("eval_mean_reward", eval_callback.last_mean_reward)

                mlflow.pytorch.log_model(
                    model.policy,
                    "model",
                    pip_requirements=["torch", "stable-baselines3", "gymnasium"],
                )
                logger.info("model_persisted_mlflow", run_id=run.info.run_id)
            except Exception as e:
                logger.error("save_failed", error=str(e))
                raise

            return {"run_id": run.info.run_id, "model_path": model_path}


def train_td3(total_timesteps: int = 10000, model_path: str = "models/best_td3"):
    trainer = RLTrainer("rl_trading_core", tracking_uri=settings.tracking_uri)
    return trainer.train_and_evaluate(total_timesteps=total_timesteps, model_path=model_path)


def train_distributed(*args, **kwargs):
    return train_td3(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Train RL Trading Policy")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--output", type=str, default="models/td3_final")
    parser.add_argument("--study_name", type=str, default="rl_trading_core")
    parser.add_argument("--tracking_uri", type=str, default=None)
    parser.add_argument("--warm_start", type=str, default=None)
    parser.add_argument("--sync_freq", type=int, default=1000)

    args = parser.parse_args()

    trainer = RLTrainer(args.study_name, tracking_uri=args.tracking_uri or settings.tracking_uri)
    trainer.train_and_evaluate(
        total_timesteps=args.timesteps,
        model_path=args.output,
        warm_start_path=args.warm_start,
        sync_freq=args.sync_freq,
    )


if __name__ == "__main__":
    main()