import pytest
import numpy as np
import torch
from src.ml.rl.augmented_agent import SentimentExtractor, AugmentedRLAgent
from unittest.mock import patch, MagicMock

def test_sentiment_extractor_logic():
    extractor = SentimentExtractor()
    text = "Safaricom shares surge after record profits announced."
    score = extractor.get_sentiment_score(text)
    assert -1 <= score <= 1
    assert score > 0

def test_sentiment_extractor_negative():
    extractor = SentimentExtractor()
    text = "Market crashes as inflation fears rise."
    score = extractor.get_sentiment_score(text)
    assert score < 0

def test_sentiment_extractor_neutral():
    extractor = SentimentExtractor()
    # Mock model output to return neutral label
    if extractor.model:
        with patch.object(extractor, 'model') as mock_model:
            mock_logits = MagicMock()
            # 0: pos, 1: neg, 2: neutral. 
            # score = probs[0] - probs[1]. If neutral is highest, we want pos/neg to be equal or small.
            mock_logits.logits = torch.tensor([[0.1, 0.1, 0.8]])
            mock_model.return_value = mock_logits
            assert extractor.get_sentiment_score("Some neutral text") == 0.0

def test_sentiment_extractor_load_failure():
    with patch("src.ml.rl.augmented_agent.AutoModelForSequenceClassification.from_pretrained", side_effect=Exception("Load fail")):
        extractor = SentimentExtractor()
        assert extractor.model is None
        assert extractor.get_sentiment_score("Any text") == 0.0

def test_sentiment_extractor_runtime_failure():
    extractor = SentimentExtractor()
    if extractor.model:
        with patch.object(extractor, 'model', side_effect=Exception("Runtime error")):
            assert extractor.get_sentiment_score("Any text") == 0.0
    else:
        assert extractor.get_sentiment_score("Any text") == 0.0

def test_augmented_rl_agent_observation():
    agent = AugmentedRLAgent(price_state_dim=5, sentiment_state_dim=1)
    price_state = np.array([100.0, 101.0, 99.0, 102.0, 105.0])
    sentiment_score = 0.8
    
    observation = agent.get_augmented_observation(price_state, sentiment_score)
    assert observation.shape == (6,)
    assert observation[-1] == 0.8
    assert np.array_equal(observation[:5], price_state)

def test_augmented_rl_agent_observation_padding():
    agent = AugmentedRLAgent(price_state_dim=5, sentiment_state_dim=1)
    # Testing truncation if input is too long
    price_state = np.array([1, 2, 3, 4, 5, 6, 7])
    sentiment_score = 0.8
    
    observation = agent.get_augmented_observation(price_state, sentiment_score)
    assert observation.shape == (6,)
    assert observation[:5].tolist() == [1, 2, 3, 4, 5]
    assert observation[5] == 0.8

@pytest.mark.asyncio
async def test_sentiment_extractor_with_llm_stub():
    extractor = SentimentExtractor()
    with pytest.raises(NotImplementedError):
        await extractor.analyze_complex_news("Federal Reserve expected to hold interest rates steady.")