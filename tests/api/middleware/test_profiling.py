import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import Request
from starlette.responses import Response, HTMLResponse
from src.api.middleware.profiling import ProfilingMiddleware

@pytest.mark.asyncio
async def test_profiling_middleware_normal_request():
    middleware = ProfilingMiddleware(app=MagicMock())
    request = MagicMock(spec=Request)
    request.query_params = {}
    
    response = Response()
    call_next = AsyncMock(return_value=response)
    
    with patch("src.config.settings") as mock_settings:
        mock_settings.DEBUG = True
        result = await middleware.dispatch(request, call_next)
        
        assert result == response
        assert "X-Process-Time-MS" in result.headers
        call_next.assert_called_once_with(request)

@pytest.mark.asyncio
async def test_profiling_middleware_slow_request():
    middleware = ProfilingMiddleware(app=MagicMock())
    request = MagicMock(spec=Request)
    request.query_params = {}
    request.method = "GET"
    request.url.path = "/test"
    
    response = Response()
    
    # Simulate a slow request
    async def slow_call_next(req):
        time.sleep(0.15) # Sleep for > 100ms
        return response
        
    with patch("src.config.settings") as mock_settings, \
         patch("src.api.middleware.profiling.logger") as mock_logger:
        mock_settings.DEBUG = True
        result = await middleware.dispatch(request, slow_call_next)
        
        assert result == response
        assert "X-Process-Time-MS" in result.headers
        mock_logger.warning.assert_called()

@pytest.mark.asyncio
async def test_profiling_middleware_with_profile_param():
    middleware = ProfilingMiddleware(app=MagicMock())
    request = MagicMock(spec=Request)
    request.query_params = {"profile": "true"}
    
    response = Response()
    call_next = AsyncMock(return_value=response)
    
    with patch("src.config.settings") as mock_settings, \
         patch("pyinstrument.Profiler") as mock_profiler_class:
        mock_settings.DEBUG = True
        mock_profiler = mock_profiler_class.return_value
        mock_profiler.output_html.return_value = "<html>Profile Report</html>"
        
        result = await middleware.dispatch(request, call_next)
        
        assert isinstance(result, HTMLResponse)
        assert result.body == b"<html>Profile Report</html>"
        mock_profiler.start.assert_called_once()
        mock_profiler.stop.assert_called_once()
        call_next.assert_called_once_with(request)

@pytest.mark.asyncio
async def test_profiling_middleware_profile_param_disabled_in_prod():
    middleware = ProfilingMiddleware(app=MagicMock())
    request = MagicMock(spec=Request)
    request.query_params = {"profile": "true"}
    
    response = Response()
    call_next = AsyncMock(return_value=response)
    
    with patch("src.config.settings") as mock_settings:
        mock_settings.DEBUG = False
        result = await middleware.dispatch(request, call_next)
        
        assert result == response
        assert "X-Process-Time-MS" in result.headers
        call_next.assert_called_once_with(request)
