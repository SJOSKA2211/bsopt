from src.utils.errors import ServiceUnavailableException, robust_pricing_task
from tests.test_utils import assert_equal


def test_robust_pricing_task_decorator():
    @robust_pricing_task(error_return_value=-1.0)
    def failing_task():
        raise ValueError("Simulated failure")

    @robust_pricing_task(error_return_value=None)
    def success_task(val):
        return val * 2

    # Test failure
    result = failing_task()
    assert_equal(result, -1.0)

    # Test success
    result = success_task(10)
    assert_equal(result, 20)


def test_service_unavailable_exception():
    exc = ServiceUnavailableException("TestService")
    assert_equal(exc.status_code, 503)
    assert "TestService" in exc.detail
