import unittest
from unittest.mock import MagicMock, patch, ANY
import json
import logging
from src.api.middleware.logging import StructuredLogger, RequestLoggingMiddleware
from fastapi import Request, Response

class TestStructuredLogger(unittest.TestCase):
    def setUp(self):
        self.logger_name = "test_logger"
        self.structured_logger = StructuredLogger(self.logger_name)
        self.mock_logger = MagicMock()
        self.structured_logger.logger = self.mock_logger

    def test_set_default_fields(self):
        self.structured_logger.set_default_fields(app="test_app")
        self.assertEqual(self.structured_logger.default_fields, {"app": "test_app"})

    def test_info_log(self):
        self.structured_logger.info("test message", key="value")
        self.mock_logger.info.assert_called_once()
        args = self.mock_logger.info.call_args[0]
        log_entry = json.loads(args[0])
        self.assertEqual(log_entry["level"], "INFO")
        self.assertEqual(log_entry["message"], "test message")
        self.assertEqual(log_entry["key"], "value")
        self.assertIn("timestamp", log_entry)

    def test_error_log(self):
        self.structured_logger.error("error message")
        self.mock_logger.error.assert_called_once()
        args = self.mock_logger.error.call_args[0]
        log_entry = json.loads(args[0])
        self.assertEqual(log_entry["level"], "ERROR")

    def test_debug_log(self):
        self.structured_logger.debug("debug message")
        self.mock_logger.debug.assert_called_once()

    def test_warning_log(self):
        self.structured_logger.warning("warning message")
        self.mock_logger.warning.assert_called_once()

    def test_critical_log(self):
        self.structured_logger.critical("critical message")
        self.mock_logger.critical.assert_called_once()

    def test_exception_log(self):
        try:
            raise ValueError("test error")
        except ValueError:
            self.structured_logger.exception("exception message")
        
        self.mock_logger.error.assert_called_once()
        args = self.mock_logger.error.call_args[0]
        log_entry = json.loads(args[0])
        self.assertIn("traceback", log_entry)
        self.assertIn("ValueError", log_entry["traceback"])

class TestRequestLoggingMiddleware(unittest.TestCase):
    def setUp(self):
        self.app = MagicMock()
        self.middleware = RequestLoggingMiddleware(self.app, persist_to_db=False)

    def test_redact_headers(self):
        headers = {
            "Authorization": "Bearer secret",
            "Cookie": "session=123",
            "Content-Type": "application/json"
        }
        redacted = self.middleware._redact_headers(headers)
        self.assertEqual(redacted["Authorization"], "[REDACTED]")
        self.assertEqual(redacted["Cookie"], "[REDACTED]")
        self.assertEqual(redacted["Content-Type"], "application/json")

    def test_redact_params(self):
        params = {
            "password": "secret",
            "token": "abc",
            "page": "1"
        }
        redacted = self.middleware._redact_params(params)
        self.assertEqual(redacted["password"], "[REDACTED]")
        self.assertEqual(redacted["token"], "[REDACTED]")
        self.assertEqual(redacted["page"], "1")

    def test_truncate_body(self):
        short_body = "short"
        self.assertEqual(self.middleware._truncate_body(short_body), short_body)
        
        # Make body significantly longer than max_length + suffix overhead
        long_body = "a" * (self.middleware.max_body_length + 1000)
        truncated = self.middleware._truncate_body(long_body)
        self.assertTrue(truncated.endswith("bytes total]"))
        self.assertTrue(len(truncated) < len(long_body))

    def test_should_skip(self):
        self.assertTrue(self.middleware._should_skip("/health"))
        self.assertFalse(self.middleware._should_skip("/api/users"))

    def test_should_reduce_log(self):
        self.assertTrue(self.middleware._should_reduce_log("/docs"))
        self.assertFalse(self.middleware._should_reduce_log("/api/users"))

class TestRequestLoggingMiddlewareIntegration(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        
        self.app = FastAPI()
        # Mock dependencies
        self.patcher_persist = patch('src.api.middleware.logging.RequestLoggingMiddleware._persist_log', new_callable=MagicMock)
        self.patcher_logger = patch('src.api.middleware.logging.request_logger')
        
        self.mock_persist = self.patcher_persist.start()
        self.mock_logger = self.patcher_logger.start()
        
        self.app.add_middleware(RequestLoggingMiddleware, persist_to_db=True, log_request_body=True)
        
        @self.app.get("/test")
        def test_route():
            return {"message": "success"}
            
        @self.app.post("/test-body")
        def test_body_route(data: dict):
            return {"received": data}
            
        @self.app.get("/error")
        def error_route():
            raise ValueError("Test Error")
            
        @self.app.get("/health")
        def health_route():
            return {"status": "ok"}

        self.client = TestClient(self.app)

    def tearDown(self):
        patch.stopall()

    def test_log_successful_request(self):
        response = self.client.get("/test")
        self.assertEqual(response.status_code, 200)
        
        # Verify log called
        self.mock_logger.log.assert_called()
        args = self.mock_logger.log.call_args
        log_entry = json.loads(args[0][1])
        
        self.assertEqual(log_entry["path"], "/test")
        self.assertEqual(log_entry["method"], "GET")
        self.assertEqual(log_entry["status_code"], 200)
        self.assertIn("duration_ms", log_entry)
        
        # Verify persistence called
        self.mock_persist.assert_called()

    def test_log_request_body(self):
        response = self.client.post("/test-body", json={"key": "value"})
        self.assertEqual(response.status_code, 200)
        
        self.mock_logger.log.assert_called()
        args = self.mock_logger.log.call_args
        log_entry = json.loads(args[0][1])
        
        self.assertIn("body", log_entry)
        self.assertIn("key", log_entry["body"])

    def test_log_error_request(self):
        with self.assertRaises(ValueError):
            self.client.get("/error")
            
        # Exception propagates, but middleware should catch, log, and re-raise
        # Wait, BaseHTTPMiddleware re-raises exceptions? Yes.
        # But we capture it in `except Exception as e` block in dispatch
        # and then `raise`.
        # So we should see an error log.
        
        self.mock_logger.log.assert_called()
        args = self.mock_logger.log.call_args
        log_entry = json.loads(args[0][1])
        
        self.assertEqual(log_entry["status_code"], 500)
        self.assertIn("error", log_entry)
        self.assertEqual(log_entry["error"]["message"], "Test Error")

    def test_skip_paths(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        
        # Should NOT log
        self.mock_logger.log.assert_not_called()
        self.mock_persist.assert_not_called()

if __name__ == '__main__':
    unittest.main()
