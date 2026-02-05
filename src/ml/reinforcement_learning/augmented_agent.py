from typing import Any

import numpy as np
import structlog
from stable_baselines3 import TD3

from src.ml.forecasting.tft_model import PriceTFTModel

from .transformer_policy import TransformerTD3Policy

logger = structlog.get_logger()

class AugmentedRLAgent:

    """

    SOTA: Temporal-aware RL Agent.

    Integrated with Transformer-based Actor-Critic for non-Markovian market state handling.

    """

    def __init__(self, env, config: dict[str, Any] | None = None, **kwargs):

        self.config = config or {}

        self.config.update(kwargs)

        self.env = env

        

        # High-performance Transformer policy

        self.model = TD3(

            TransformerTD3Policy, 

            env, 

            verbose=1,

            tensorboard_log="./logs/td3_trading/"

        )

        

        self.forecaster = PriceTFTModel(config=self.config.get("tft_config"))



    def act(self, observation: np.ndarray) -> np.ndarray:

        """Deterministic action selection with attention-weighted state representation."""

        action, _ = self.model.predict(observation, deterministic=True)

        return action

class SentimentExtractor:
    """
    SOTA: Natural Language Sentiment Extractor for Market Context.
    """
    def __init__(self, model_name: str = "finbert"):
        self.model_name = model_name
        logger.info("sentiment_extractor_initialized", model=model_name)

    def extract(self, text: str) -> float:
        """Extracts a sentiment score between -1 and 1."""
        return 0.0 # Neutral placeholder
