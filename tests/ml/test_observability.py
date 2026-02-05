import json

import pytest
import structlog

from src.shared.observability import SCRAPE_DURATION, SCRAPE_ERRORS, setup_logging


def test_setup_logging_json(capsys):
    """Verify that logging is configured to output JSON."""
    setup_logging()
    logger = structlog.get_logger("test_logger")
    
    logger.info("test_message", key="value")
    
    captured = capsys.readouterr()
    log_output = captured.out.strip()
    
    # Try to parse as JSON
    try:
        log_json = json.loads(log_output)
        assert log_json["event"] == "test_message"
        assert log_json["key"] == "value"
        assert "timestamp" in log_json
        assert log_json["level"] == "info"
    except json.JSONDecodeError:
        pytest.fail("Log output is not valid JSON")

def test_metrics_definitions():
    """Verify that metrics are defined with correct labels."""
    assert SCRAPE_DURATION._labelnames == ('api',)
    assert SCRAPE_ERRORS._labelnames == ('api', 'status_code')
