import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.ml.pipelines.sentiment_ingest import SentimentIngestor
import json

@pytest.mark.asyncio
async def test_sentiment_ingestor_process_message():
    with patch("src.ml.pipelines.sentiment_ingest.SentimentExtractor") as mock_extractor:
        with patch("src.ml.pipelines.sentiment_ingest.Producer") as mock_producer:
            mock_extractor.return_value.get_sentiment_score.return_value = 0.75
            ingestor = SentimentIngestor(bootstrap_servers="localhost:9092")
            
            msg_data = {"symbol": "AAPL", "text": "Apple is great.", "timestamp": "2026"}
            await ingestor.process_news_message(json.dumps(msg_data).encode('utf-8'))
            
            mock_extractor.return_value.get_sentiment_score.assert_called_with("Apple is great.")
            assert mock_producer.return_value.produce.called

@pytest.mark.asyncio
async def test_process_empty_message():
    with patch("src.ml.pipelines.sentiment_ingest.SentimentExtractor") as mock_extractor:
        ingestor = SentimentIngestor()
        await ingestor.process_news_message(json.dumps({"text": ""}).encode('utf-8'))
        assert not mock_extractor.return_value.get_sentiment_score.called

@pytest.mark.asyncio
async def test_process_invalid_json():
    ingestor = SentimentIngestor()
    # Should not raise exception
    await ingestor.process_news_message(b"invalid json")

def test_sentiment_ingestor_run_loop():
    with patch("src.ml.pipelines.sentiment_ingest.SentimentExtractor"):
        with patch("src.ml.pipelines.sentiment_ingest.Producer"):
            with patch("src.ml.pipelines.sentiment_ingest.Consumer") as mock_cons:
                ingestor = SentimentIngestor()
                # Mock poll to return None once then raise exception to break loop
                mock_cons.return_value.poll.side_effect = [None, Exception("Stop loop")]
                
                with pytest.raises(Exception, match="Stop loop"):
                    ingestor.run()
                
                assert mock_cons.return_value.subscribe.called
                assert mock_cons.return_value.close.called

def test_sentiment_ingestor_run_kafka_error():
    with patch("src.ml.pipelines.sentiment_ingest.SentimentExtractor"):
        with patch("src.ml.pipelines.sentiment_ingest.Producer"):
            with patch("src.ml.pipelines.sentiment_ingest.Consumer") as mock_cons:
                ingestor = SentimentIngestor()
                # Mock poll to return a message with an error, then stop
                error_msg = MagicMock()
                error_msg.error.return_value = True
                mock_cons.return_value.poll.side_effect = [error_msg, Exception("Stop loop")]
                
                with pytest.raises(Exception, match="Stop loop"):
                    ingestor.run()
                
                assert mock_cons.return_value.poll.called
