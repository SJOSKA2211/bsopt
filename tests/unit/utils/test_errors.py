from src.utils.errors import ServiceUnavailableException, robust_pricing_task


def test_robust_pricing_task_decorator():
    @robust_pricing_task(error_return_value={"error": True})
    def failing_task():
        raise ValueError("Something went wrong")

    @robust_pricing_task(error_return_value=None)
    def success_task():
        return 42

    assert failing_task() == {"error": True}
    assert success_task() == 42


def test_service_unavailable_exception():
    exc = ServiceUnavailableException("Redis")
    assert exc.status_code == 503
    assert "Redis" in exc.detail
