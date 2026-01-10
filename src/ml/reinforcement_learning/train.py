import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
import mlflow
import structlog
from src.ml.reinforcement_learning.trading_env import TradingEnvironment

logger = structlog.get_logger()

def train_td3(total_timesteps: int = 100000, model_path: str = "models/td3_trading_agent"):
    """
    Train a TD3 agent on the custom TradingEnvironment.
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
            "policy": "MlpPolicy",
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
            eval_freq=5000,
            deterministic=True, 
            render=False
        )
        
        # Train the agent
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        
        # Save the model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        
        # Log model to MLflow
        mlflow.pytorch.log_model(model.policy, "model")
        
        logger.info("training_completed", run_id=run.info.run_id, model_path=model_path)
        
    return model

def main():
    parser = argparse.ArgumentParser(description="Train RL Trading Agent")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total training timesteps")
    parser.add_argument("--output", type=str, default="models/td3_trading_agent", help="Path to save the model")
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    train_td3(total_timesteps=args.timesteps, model_path=args.output)

if __name__ == "__main__":
    main()
