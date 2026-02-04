import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from src.ml.pipelines.sentiment_ingest import SentimentPipeline

@pytest.fixture
def mock_extractor():
    with patch("src.ml.pipelines.sentiment_ingest.SentimentExtractor") as mock:
        instance = mock.return_value
        instance.get_sentiment_score.return_value = 0.5
        yield instance

@pytest.mark.asyncio
async def test_sentiment_pipeline_processing(mock_extractor):
    """Test that the pipeline correctly extracts sentiment from scraper messages."""
    pipeline = SentimentPipeline()
    
    # Mock message from scraper (e.g., via Kafka or direct call)
    scraper_data = {
        "text": "Market looks bullish today.",
        "symbol": "AAPL",
        "timestamp": "2026-01-14T10:00:00Z"
    }
    
    # Process message
    result = await pipeline.process_scraper_message(scraper_data)
    
    assert result["symbol"] == "AAPL"
    assert result["sentiment"] == 0.5
    mock_extractor.get_sentiment_score.assert_called_once_with("Market looks bullish today.")

@pytest.mark.asyncio
async def test_sentiment_pipeline_empty_text():
    """Test that pipeline handles empty text gracefully."""
    pipeline = SentimentPipeline()
    result = await pipeline.process_scraper_message({"text": "", "symbol": "AAPL"})
    assert result["sentiment"] == 0.0

@pytest.mark.asyncio
async def test_sentiment_pipeline_error_handling(mock_extractor):
    """Test that pipeline handles extractor errors gracefully."""
    pipeline = SentimentPipeline()
    mock_extractor.get_sentiment_score.side_effect = Exception("Extraction failed")
    
    result = await pipeline.process_scraper_message({"text": "faulty", "symbol": "AAPL"})
    assert result["sentiment"] == 0.0

def test_sentiment_pipeline_initialization():
    """Verify pipeline initializes its components."""
    pipeline = SentimentPipeline()
    assert pipeline.extractor is not None
