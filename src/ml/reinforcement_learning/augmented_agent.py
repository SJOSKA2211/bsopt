from typing import Any

import numpy as np
import structlog
from stable_baselines3 import TD3

from src.ml.forecasting.tft_model import PriceTFTModel

from .transformer_policy import TransformerTD3Policy

logger = structlog.get_logger()


class AugmentedRLAgent:
    """
    OPTIMIZED: Multimodal RL Agent.
    Fuses Price Forecasts (TFT) and NLP Sentiment into the observation window.
    """

    def __init__(self, env, config: dict[str, Any] | None = None, **kwargs):
        self.config = config or {}
        self.config.update(kwargs)
        self.env = env

        self.model = TD3(
            TransformerTD3Policy, env, verbose=1, tensorboard_log="./logs/td3_trading/"
        )

        self.forecaster = PriceTFTModel(config=self.config.get("tft_config"))
        self.sentiment_extractor = SentimentExtractor()

    async def act(self, observation: np.ndarray, news_text: str | None = None) -> np.ndarray:
        """
        Multimodal action selection with ACTUAL Observation Fusion.
        1. Extract sentiment from news.
        2. Get forecast from TFT.
        3. FUSE: Concatenate forecast/sentiment into observation.
        """
        sentiment = 0.0
        if news_text:
            sentiment = self.sentiment_extractor.extract(news_text)

        # Institutional Inference: High-performance call to TFT forecaster core
        forecast = self.forecaster.predict(observation)
        if isinstance(forecast, np.ndarray):
            # Take the mean forecast across lookahead if it's a sequence
            forecast_val = np.mean(forecast)
        else:
            forecast_val = float(forecast or 0.0)

        # FUSION: Append [forecast, sentiment] to the observation
        # Assuming observation is (Batch, Features) or (Features,)
        if observation.ndim == 1:
            augmented_obs = np.append(observation, [forecast_val, sentiment])
        else:
            # Append to every sample in batch
            batch_size = observation.shape[0]
            extra = np.tile([forecast_val, sentiment], (batch_size, 1))
            augmented_obs = np.hstack([observation, extra])

        logger.debug("multimodal_fusion_complete", sentiment=sentiment, forecast=forecast_val)

        action, _ = self.model.predict(augmented_obs, deterministic=True)
        return action


class SentimentExtractor:
    """
    High-Performance Lexicon-based Extractor.
    Uses O(1) hash-based lookup for institutional sentiment signals.
    """

    def __init__(self, model_name: str = "finbert"):
        self.model_name = model_name
        # Pre-process keywords into a set for O(1) lookup
        self.keyword_map = {
            "bullish": 0.8,
            "bearish": -0.8,
            "upgraded": 0.5,
            "downgraded": -0.5,
            "beat": 0.4,
            "missed": -0.4,
            "profit": 0.2,
            "loss": -0.3,
        }

    def extract(self, text: str) -> float:
        """Extracts a sentiment score using optimized lookup."""
        if not text:
            return 0.0

        import re

        # OPTIMIZATION: Use regex to tokenize properly while ignoring punctuation
        words = re.findall(r"\w+", text.lower())

        score = 0.0
        matches = 0

        for word in words:
            if word in self.keyword_map:
                score += self.keyword_map[word]
                matches += 1

        return np.clip(score / max(matches, 1), -1.0, 1.0) if matches > 0 else 0.0

    def get_sentiment_score(self, text: str) -> float:
        """Alias for extract() for backward compatibility."""
        return self.extract(text)
