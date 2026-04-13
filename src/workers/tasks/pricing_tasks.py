import numpy as np

from src.workers.tasks.celery_app import celery_app


@celery_app.task
def price_option_task(spot, strike, maturity, volatility, rate, option_type="call"):
    """Task to price a single option."""
    from src.math_kernel.black_scholes import BlackScholesEngine
    return float(BlackScholesEngine.price_options(
        spot=spot, strike=strike, maturity=maturity, 
        volatility=volatility, rate=rate, option_type=option_type
    ))

@celery_app.task
def batch_price_options_task(spots, strikes, maturities, volatilities, rates, option_types):
    """Task to price a batch of options."""
    from src.math_kernel.black_scholes import BlackScholesEngine
    # price_batch expects numpy arrays
    return BlackScholesEngine.price_batch(
        np.array(spots), np.array(strikes), np.array(maturities), 
        np.array(volatilities), np.array(rates), 
        np.zeros_like(spots), np.array(option_types)
    ).tolist()

@celery_app.task
def generate_volatility_surface_task(symbol):
    """Task to generate a volatility surface."""
    return {"symbol": symbol, "surface": []}