import numpy as np

from src.shared.math_utils import calculate_price


def generate_synthetic_data_numba(
    n_samples: int = 10000, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Generate synthetic training data using NumPy-optimized Black-Scholes engine.
    """
    np.random.seed(random_state)

    S = np.random.uniform(50, 150, n_samples)
    K = np.random.uniform(50, 150, n_samples)
    T = np.random.uniform(0.1, 2.0, n_samples)
    r = np.random.uniform(0.01, 0.05, n_samples)
    sigma = np.random.uniform(0.1, 0.5, n_samples)
    is_call_int = np.random.choice([0, 1], n_samples)
    is_call = is_call_int.astype(bool)

    # Use shared math utils - vectorized NumPy calculation
    q = np.zeros_like(S)
    prices = calculate_price(S, K, T, sigma, r, q, is_call)

    # Construct features
    X = np.column_stack(
        [S, K, T, is_call_int, S / K, np.log(S / K), np.sqrt(T), T * 365, sigma]
    )

    feature_names = [
        "underlying_price",
        "strike",
        "time_to_expiry",
        "is_call",
        "moneyness",
        "log_moneyness",
        "sqrt_time_to_expiry",
        "days_to_expiry",
        "implied_volatility",
    ]
    return X, prices, feature_names
