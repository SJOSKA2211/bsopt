import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.tasks.pricing_tasks import (
    price_option_task, 
    batch_price_options_task, 
    calculate_implied_volatility_task, 
    generate_volatility_surface_task
)

@pytest.fixture
def mock_cache():
    with patch("src.utils.cache.pricing_cache", new_callable=MagicMock) as mock_pc:
        import asyncio
        async def async_return(val):
            return val
            
        mock_pc.get_option_price.side_effect = lambda *args, **kwargs: async_return(None)
        mock_pc.get_greeks.side_effect = lambda *args, **kwargs: async_return(None)
        mock_pc.set_option_price.side_effect = lambda *args, **kwargs: async_return(None)
        mock_pc.set_greeks.side_effect = lambda *args, **kwargs: async_return(None)
        yield mock_pc

@pytest.fixture
def mock_task_context():
    mock_self = MagicMock()
    mock_self.request.id = "test-task-id"
    return mock_self

def test_price_option_task_no_cache(mock_task_context, mock_cache):
    from src.tasks.celery_app import celery_app
    celery_app.conf.task_always_eager = True
    
    res = price_option_task.apply(kwargs={
        "spot": 100.0, "strike": 100.0, "maturity": 1.0, 
        "volatility": 0.2, "rate": 0.05, "use_cache": False
    })
    
    result = res.get()
    assert result["price"] > 0
    assert result["status"] == "completed"
    assert result["cache_hit"] is False

def test_price_option_task_logic():
    from src.tasks.celery_app import celery_app
    celery_app.conf.task_always_eager = True
    
    res = price_option_task.apply(args=[], kwargs={
        "spot": 100.0, "strike": 100.0, "maturity": 1.0, 
        "volatility": 0.2, "rate": 0.05, "use_cache": False
    })
    
    result = res.get()
    assert result["price"] > 0
    assert result["status"] == "completed"

def test_price_option_task_with_cache(mock_cache):
    import asyncio
    mock_loop = MagicMock()
    
    # Mock cache hit
    cached_price = 10.5
    mock_greeks = MagicMock()
    mock_greeks.delta = 0.5
    mock_greeks.gamma = 0.1
    mock_greeks.vega = 0.2
    mock_greeks.theta = -0.1
    mock_greeks.rho = 0.1
    
    # pricing_cache.get_option_price returns a coroutine normally, 
    # but since we mocked the methods to return a coroutine in mock_cache fixture,
    # run_until_complete will receive that coroutine.
    # We need mock_loop.run_until_complete to return our desired values.
    mock_loop.run_until_complete.side_effect = [cached_price, mock_greeks, None, None]

    with patch("asyncio.get_event_loop", return_value=mock_loop):
        from src.tasks.celery_app import celery_app
        celery_app.conf.task_always_eager = True
        
        res = price_option_task.apply(kwargs={
            "spot": 100.0, "strike": 100.0, "maturity": 1.0, 
            "volatility": 0.2, "rate": 0.05, "use_cache": True
        })
        result = res.get()
        assert result["cache_hit"] is True
        assert result["price"] == 10.5

def test_batch_price_options_task():
    options = [
        {"spot": 100, "strike": 100, "maturity": 1, "volatility": 0.2, "rate": 0.05},
        {"spot": 110, "strike": 100, "maturity": 1, "volatility": 0.2, "rate": 0.05}
    ] * 6 
    
    from src.tasks.celery_app import celery_app
    celery_app.conf.task_always_eager = True
    
    res = batch_price_options_task.apply(kwargs={"options": options, "use_vectorized": True})
    result = res.get()
    assert result["count"] == 12
    assert result["vectorized"] is True
    assert len(result["results"]) == 12

def test_calculate_implied_volatility_task():
    market_price = 10.4506
    
    from src.tasks.celery_app import celery_app
    celery_app.conf.task_always_eager = True
    
    res = calculate_implied_volatility_task.apply(kwargs={
        "market_price": market_price,
        "spot": 100.0, "strike": 100.0, "maturity": 1.0, "rate": 0.05
    })
    result = res.get()
    assert result["converged"] is True
    assert 0.19 < result["implied_volatility"] < 0.21

def test_generate_volatility_surface_task():
    strikes = [90, 100, 110]
    maturities = [0.5, 1.0, 2.0]
    
    from src.tasks.celery_app import celery_app
    celery_app.conf.task_always_eager = True
    
    res = generate_volatility_surface_task.apply(kwargs={
        "spot": 100.0, "strikes": strikes, "maturities": maturities, "rate": 0.05
    })
    result = res.get()
    surface = result["surface"]
    assert len(surface) == 3 
    assert len(surface[0]) == 3
    assert surface[0][0] > 0

def test_batch_price_options_small_batch():
    options = [{"spot": 100, "strike": 100, "maturity": 1, "volatility": 0.2, "rate": 0.05}]
    
    from src.tasks.celery_app import celery_app
    celery_app.conf.task_always_eager = True
    
    res = batch_price_options_task.apply(kwargs={"options": options, "use_vectorized": False})
    result = res.get()
    assert result["count"] == 1
    assert result["results"][0]["price"] > 0
    assert result["vectorized"] is False

def test_price_option_task_input_validation():
    from src.tasks.celery_app import celery_app
    celery_app.conf.task_always_eager = True
    
    with pytest.raises(ValueError):
        price_option_task.apply(kwargs={"spot": -100.0, "strike": 100.0, "maturity": 1.0, "volatility": 0.2, "rate": 0.05}).get()
        
    with pytest.raises(ValueError):
        price_option_task.apply(kwargs={"spot": 100.0, "strike": 100.0, "maturity": 1.0, "volatility": 0.2, "rate": 0.05, "option_type": "invalid"}).get()

def test_cache_exception_handling(mock_cache):
    import asyncio
    async def async_raise(*args, **kwargs):
        raise Exception("Redis down")
    
    mock_cache.get_option_price.side_effect = lambda *args, **kwargs: async_raise()
    mock_cache.get_greeks.side_effect = lambda *args, **kwargs: async_raise()
    
    from src.tasks.celery_app import celery_app
    celery_app.conf.task_always_eager = True
    
    res = price_option_task.apply(kwargs={"spot": 100.0, "strike": 100.0, "maturity": 1.0, "volatility": 0.2, "rate": 0.05, "use_cache": True})
    assert res.get()["status"] == "completed"
    
    async def async_return(val):
        return val
        
    mock_cache.get_option_price.side_effect = lambda *args, **kwargs: async_return(None)
    mock_cache.set_option_price.side_effect = lambda *args, **kwargs: async_raise()
    
    res = price_option_task.apply(kwargs={"spot": 100.0, "strike": 100.0, "maturity": 1.0, "volatility": 0.2, "rate": 0.05, "use_cache": True})
    assert res.get()["status"] == "completed"

def test_iv_task_validation():
    from src.tasks.celery_app import celery_app
    celery_app.conf.task_always_eager = True
    
    try:
        calculate_implied_volatility_task.apply(kwargs={
            "market_price": 5.0,
            "spot": 100.0, "strike": 300.0, "maturity": 0.01, "rate": 0.05
        }).get()
    except Exception:
        pass 

def test_pricing_task_general_exception():
    with patch("src.tasks.pricing_tasks.BlackScholesEngine.price_options", side_effect=Exception("Math Error")):
        from src.tasks.celery_app import celery_app
        celery_app.conf.task_always_eager = True
        
        with pytest.raises(Exception):
            price_option_task.apply(kwargs={"spot": 100.0, "strike": 100.0, "maturity": 1.0, "volatility": 0.2, "rate": 0.05}).get()

def test_batch_pricing_error():
    # Use > 10 items to trigger vectorized path
    options = [{"spot": 100}] * 11
    with patch("src.tasks.pricing_tasks.BlackScholesEngine.price_options", side_effect=Exception("Batch Fail")):
        from src.tasks.celery_app import celery_app
        celery_app.conf.task_always_eager = True
        
        with pytest.raises(Exception):
            batch_price_options_task.apply(kwargs={"options": options, "use_vectorized": True}).get()

def test_vol_surface_error():
    from src.tasks.celery_app import celery_app
    celery_app.conf.task_always_eager = True
    
    # Trigger math domain error (log of negative/zero)
    # spot=0
    with pytest.raises(Exception):
        generate_volatility_surface_task.apply(kwargs={
            "spot": -100.0, "strikes": [100], "maturities": [1.0], "rate": 0.05
        }).get()
