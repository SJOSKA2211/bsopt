import pytest
import torch
from unittest.mock import MagicMock, patch
from src.ml.rl.augmented_agent import SentimentExtractor

@pytest.fixture
def mock_transformers():
    """Mock transformers library to avoid downloading large models during tests."""
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_from_pretrained, \
         patch("transformers.AutoModelForSequenceClassification.from_pretrained") as mock_model_from_pretrained:
        
        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
        
        # Mock model
        mock_model_instance = MagicMock()
        mock_model_from_pretrained.return_value = mock_model_instance
        
        # Mock model output
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[10.0, 1.0, 1.0]]) 
        mock_model_instance.return_value = mock_output
        
        yield mock_tokenizer_instance, mock_model_instance

def test_sentiment_extractor_initialization(mock_transformers):
    """Test that SentimentExtractor initializes with the correct model."""
    extractor = SentimentExtractor(model_name="ProsusAI/finbert")
    assert extractor.model_name == "ProsusAI/finbert"
    assert extractor.tokenizer is not None
    assert extractor.model is not None

def test_extract_sentiment_score(mock_transformers):
    """Test that extractor returns a normalized sentiment score between -1 and 1."""
    mock_tokenizer, mock_model = mock_transformers
    extractor = SentimentExtractor()
    
    # Force mock output for single call: [Pos, Neg, Neu]
    # Probs will be softmax of logits.
    # To get high positive score (probs[0] - probs[1]), we need high first logit.
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([[10.0, 1.0, 1.0]]) # 10.0 will lead to ~1.0 prob after softmax
    mock_model.return_value = mock_output
    
    text = "Stocks are soaring."
    score = extractor.get_sentiment_score(text)
    
    assert score > 0.8
    assert -1 <= score <= 1

def test_sentiment_extractor_load_failure():
    """Test SentimentExtractor behavior when model fails to load."""
    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=Exception("Load failed")):
        extractor = SentimentExtractor()
        assert extractor.model is None
        assert extractor.get_sentiment_score("test") == 0.0
        assert extractor.get_sentiment_scores_batch(["test"]) == [0.0]

def test_extract_sentiment_batch(mock_transformers):
    """Test batch sentiment extraction."""
    mock_tokenizer, mock_model = mock_transformers
    extractor = SentimentExtractor()
    texts = ["Good news!", "Bad news.", "Neutral."]
    
    # Mocking different outputs for batch
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([
        [0.8, 0.1, 0.1], # Pos
        [0.1, 0.8, 0.1], # Neg
        [0.1, 0.1, 0.8]  # Neu
    ])
    mock_model.return_value = mock_output
    
    # Fix the tokenizer mock for batch input
    mock_tokenizer.return_value = {"input_ids": torch.randn(3, 10), "attention_mask": torch.ones(3, 10)}
    
    scores = extractor.get_sentiment_scores_batch(texts)
    
    assert len(scores) == 3
    assert scores[0] > 0
    assert scores[1] < 0
    assert abs(scores[2]) < 0.1
