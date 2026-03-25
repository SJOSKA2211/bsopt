import os
from typing import Any

import numpy as np
import ray
import structlog
import torch

from src.ml.reinforcement_learning.train import RLTrainer

logger = structlog.get_logger()

@ray.remote(num_cpus=1, num_gpus=0)
class RolloutWorker:
    """Distributed worker for gathering trajectories using the current policy."""

    def __init__(self, env_config: dict[str, Any]):
        from src.ml.reinforcement_learning.trading_env import TradingEnvironment
        from src.ml.reinforcement_learning.transformer_policy import TransformerTD3Policy
        from src.ml.reinforcement_learning.transformer_policy import TransformerFeatureExtractor

        self.env = TradingEnvironment(**env_config)
        self.device = torch.device("cpu")
        
        # Consistent Production policy initialization
        policy_kwargs = dict(
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[256, 256], qf=[256, 256]),
        )
        
        self.policy = TransformerTD3Policy(
            self.env.observation_space,
            self.env.action_space,
            lr_schedule=lambda _: 1e-4,
            **policy_kwargs
        ).to(self.device)

    def gather_experience(self, weights: dict[str, Any], num_steps: int = 1000):
        """Gather trajectories natively mapping active model weights along episode walks."""
        if weights:
            # Load weights into policy safely
            self.policy.load_state_dict({k: torch.as_tensor(v) for k, v in weights.items()})
            self.policy.eval()

        trajectories = []
        obs, info = self.env.reset()
        
        for _ in range(num_steps):
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs).float().unsqueeze(0).to(self.device)
                action, _ = self.policy.predict(obs_tensor, deterministic=False)
            
            res = self.env.step(action)
            if len(res) == 5:
                next_obs, reward, done, truncated, info = res
            else:
                next_obs, reward, done, info = res
                truncated = False
            
            trajectories.append((obs, action, reward, next_obs, done or truncated))
            
            if done or truncated:
                obs, info = self.env.reset()
            else:
                obs = next_obs
                
        return trajectories

class RayRLTrainer:
    """
    Phase 4: Multi-Node Distributed RL Trainer using Ray.
    Orchestrates rollout workers and handles gradient updates across the cluster.
    """

    def __init__(self, num_workers: int = 4):
        if not ray.is_initialized():
            ray.init(address="auto", ignore_reinit_error=True)
        self.num_workers = num_workers
        self.workers = [RolloutWorker.remote({}) for _ in range(num_workers)]
        
        # Master trainer on head node
        self.master = RLTrainer("ray_distributed_core")
        self.model = None # Initialized on first train attempt

    def train_distributed(self, total_timesteps: int = 100000):
        """Execute distributed training loop with real weight syncing and trajectory feeding."""
        logger.info("ray_distributed_training_started", workers=self.num_workers)

        from stable_baselines3 import TD3
        from src.ml.reinforcement_learning.transformer_policy import TransformerTD3Policy
        from src.ml.reinforcement_learning.transformer_policy import TransformerFeatureExtractor
        from src.ml.reinforcement_learning.trading_env import TradingEnvironment
        
        env = TradingEnvironment()
        policy_kwargs = dict(
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[256, 256], qf=[256, 256]),
        )
        self.model = TD3(TransformerTD3Policy, env, policy_kwargs=policy_kwargs, verbose=0)

        steps_done = 0
        while steps_done < total_timesteps:
            # 1. Sync weights
            with torch.no_grad():
                active_weights = {
                    k: v.cpu().numpy() for k, v in self.model.policy.state_dict().items()
                }
            
            # 2. Gather distributed experience
            worker_tasks = [w.gather_experience.remote(weights=active_weights, num_steps=512) for w in self.workers]
            results: list[list[tuple]] = ray.get(worker_tasks)

            for trajectory in results:
                for obs, action, reward, next_obs, done in trajectory:
                    self.model.replay_buffer.add(obs, next_obs, action, reward, done, [{}])
                    steps_done += 1

            # 4. Trigger Production training step
            if self.model.replay_buffer.size() > self.model.learning_starts:
                self.model.train(batch_size=self.model.batch_size, gradient_steps=batch_samples // 64)
            
            avg_reward = np.mean([sum(t[2] for t in traj) for traj in results])
            
            logger.info(
                "distributed_training_step",
                steps=steps_done,
                avg_episode_reward=avg_reward,
                buffer_size=self.model.replay_buffer.size(),
            )

        logger.info("ray_distributed_training_complete", total_steps=steps_done)
        return {"status": "success", "steps": steps_done}

def start_distributed_training():
    """Entry point for Phase 4 distributed revamp."""
    trainer = RayRLTrainer(num_workers=int(os.getenv("RAY_WORKERS", 2)))
    return trainer.train_distributed()
