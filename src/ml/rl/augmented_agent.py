import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any, Optional
import numpy as np
import structlog

logger = structlog.get_logger()

class SentimentExtractor:
    """
    Sentiment Oracle using FinBERT for extracting market sentiment from text.
    Normalized score returned in range [-1, 1].
    """
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
        except Exception as e:
            logger.warning("sentiment_model_load_failed", error=str(e))
            self.tokenizer = None
            self.model = None

    def get_sentiment_score(self, text: str) -> float:
        """
        Extracts sentiment score from a single text string.
        
        Args:
            text (str): Input financial text.
            
        Returns:
            float: Normalized sentiment score (-1 to 1).
        """
        if self.model is None or self.tokenizer is None:
            return 0.0

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # FinBERT labels: 0: positive, 1: negative, 2: neutral
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
        
        # Pos - Neg = Score
        score = probs[0] - probs[1]
        return float(score)

    def get_sentiment_scores_batch(self, texts: List[str]) -> List[float]:
        """
        Extracts sentiment scores for a batch of texts.
        """
        if self.model is None or self.tokenizer is None:
            return [0.0] * len(texts)

        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()
        # probs[:, 0] is positive, probs[:, 1] is negative
        scores = probs[:, 0] - probs[:, 1]
        return scores.tolist()

    async def analyze_complex_news(self, text: str):
        """
        Placeholder for complex LLM-based sentiment analysis.
        """
        raise NotImplementedError("LLM integration not yet implemented")

class AugmentedRLAgent:
    """
    RL Agent enhanced with Sentiment State for comprehensive market understanding.
    
    This agent extends traditional RL by incorporating unstructured news sentiment 
    alongside structured price data into its observation space.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initializes the agent with observation and action dimensions.
        """
        self.config = config or {}
        # Merge kwargs into config if provided
        self.config.update(kwargs)
        
        self.price_state_dim = self.config.get("price_state_dim", 10)
        self.sentiment_state_dim = self.config.get("sentiment_state_dim", 1)
        self.observation_dim = self.price_state_dim + self.sentiment_state_dim
        self.action_dim = self.config.get("action_dim", 3)
        self.model = None

    def get_augmented_observation(self, price_state: np.ndarray, sentiment_score: float) -> np.ndarray:
        """
        Concatenates structured price state with unstructured sentiment score.
        
        Args:
            price_state (np.ndarray): Array of price-related features.
            sentiment_score (float): Normalized sentiment score from SentimentExtractor.
            
        Returns:
            np.ndarray: Concatenated observation [Price_State, Sentiment_State].
        """
        # Ensure price_state is a numpy array
        price_state = np.array(price_state)
        # Create sentiment array (can be more complex if sentiment_state_dim > 1)
        sentiment_state = np.array([sentiment_score])
        
        return np.concatenate([price_state, sentiment_state])

    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Produces an action based on the augmented observation.
        """
        if self.model is None:
            # Random action if no model is loaded
            return np.random.uniform(-1, 1, size=(self.action_dim,))
        
        action, _states = self.model.predict(observation, deterministic=True)
        return action