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
    High-Performance Transformer-based Sentiment Extractor.
    Uses FinBERT (ProsusAI/finbert) for institutional-grade financial sentiment analysis.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self._pipeline = None
        logger.info("sentiment_extractor_initialized", model=model_name)

    def _get_pipeline(self):
        """Lazy-load the transformer pipeline to save memory if not used."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
                self._pipeline = pipeline("sentiment-analysis", model=self.model_name)
            except ImportError:
                logger.error("transformers_not_installed", action="falling_back_to_lexicon")
                return None
        return self._pipeline

    def extract(self, text: str) -> float:
        """Extracts a sentiment score using FinBERT or optimized lexicon fallback."""
        if not text:
            return 0.0

        pipe = self._get_pipeline()
        if pipe:
            try:
                # FinBERT returns: labels (positive, negative, neutral) and scores
                result = pipe(text[:512])[0] # Truncate to model max length
                label = result["label"].lower()
                score = result["score"]
                
                if label == "positive":
                    return float(score)
                elif label == "negative":
                    return -float(score)
                return 0.0
            except Exception as e:
                logger.warning("transformer_inference_failed", error=str(e))

        # --- OPTIMIZED LEXICON FALLBACK ---
        keyword_map = {
             "bullish": 0.8, "bearish": -0.8, "upgraded": 0.5, "downgraded": -0.5,
             "beat": 0.4, "missed": -0.4, "profit": 0.2, "loss": -0.3,
             "buy": 0.3, "sell": -0.3, "growth": 0.2, "debt": -0.2
        }
        import re
        words = re.findall(r"\w+", text.lower())
        score, matches = 0.0, 0
        for word in words:
            if word in keyword_map:
                score += keyword_map[word]
                matches += 1
        return np.clip(score / max(matches, 1), -1.0, 1.0) if matches > 0 else 0.0

    def get_sentiment_score(self, text: str) -> float:
        """Alias for extract() for backward compatibility."""
        return self.extract(text)
