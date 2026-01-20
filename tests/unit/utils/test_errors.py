import pytest
from src.utils.errors import robust_pricing_task, ServiceUnavailableException
from fastapi import status

def test_robust_pricing_task_success():
    @robust_pricing_task(error_return_value=-1)
    def success_func():
        return 100
    
    assert success_func() == 100

def test_robust_pricing_task_failure(mocker):
    # Mock logger to verify error logging
    mock_logger = mocker.patch("src.utils.errors.logger")
    
    @robust_pricing_task(error_return_value=-1)
    def fail_func():
        raise ValueError("Simulated crash")
    
    result = fail_func()
    assert result == -1
    assert mock_logger.error.called

def test_service_unavailable_exception():
    exc = ServiceUnavailableException("PricingService")
    assert exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "PricingService" in exc.detail