import re

with open("tests/unit/api/middleware/test_logging_middleware_unit.py", "r") as f:
    content = f.read()

# Fix mock log to use .info instead of .log
content = content.replace('patch("api.middleware.logging.request_logger.log")', 'patch("api.middleware.logging.request_logger.info")')

# Fix json.loads call arg
content = content.replace('json.loads(mock_log.call_args[0][1])', 'json.loads(mock_log.call_args[0][0])')

with open("tests/unit/api/middleware/test_logging_middleware_unit.py", "w") as f:
    f.write(content)

with open("tests/unit/api/middleware/test_logging_middleware.py", "r") as f:
    content = f.read()

# Replace test_persist_log_full with a direct test and a middleware test
new_test = """
@pytest.mark.asyncio
@patch("src.database.SessionLocal")
async def test_persist_log_full(mock_session_local):
    with patch("src.database.models.RequestLog"):
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        middleware = RequestLoggingMiddleware(MagicMock(), persist_to_db=True)
        log_entry = {
            "request_id": "test",
            "method": "GET",
            "path": "/test",
            "status_code": 200,
            "duration_ms": 10,
            "client_ip": "127.0.0.1",
            "user_id": str(uuid.uuid4())
        }
        
        await middleware._persist_log(log_entry, MagicMock())
        
        mock_session.add.assert_called()
        mock_session.commit.assert_called()
"""
content = re.sub(r'@patch\("src\.database\.SessionLocal"\)\ndef test_persist_log_full\(.*?\n(?:    .*\n)*', new_test, content, flags=re.MULTILINE)
with open("tests/unit/api/middleware/test_logging_middleware.py", "w") as f:
    f.write("import pytest\n" + content)

with open("tests/unit/api/middleware/test_logging_middleware_unit.py", "r") as f:
    content = f.read()
content = re.sub(r'@patch\("src\.database\.get_session"\)\ndef test_persist_log_full\(.*?\n(?:    .*\n)*', new_test, content, flags=re.MULTILINE)
with open("tests/unit/api/middleware/test_logging_middleware_unit.py", "w") as f:
    f.write("import pytest\n" + content)

