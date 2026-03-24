import cupy as cp

def gpu_black_scholes(
    s: cp.ndarray, k: cp.ndarray, t: cp.ndarray, v: cp.ndarray, r: cp.ndarray, is_call: cp.ndarray
) -> cp.ndarray:
    """
    GPU-accelerated Black-Scholes pricing kernel using CuPy.
    """
    d1 = (cp.log(s / k) + (r + 0.5 * v * v) * t) / (v * cp.sqrt(t))
    d2 = d1 - v * cp.sqrt(t)

    # Normal CDF approximation for GPU
    def norm_cdf(x):
        return 0.5 * (1.0 + cp.erf(x / cp.sqrt(2.0)))

    if is_call.dtype == bool:
        price_call = s * norm_cdf(d1) - k * cp.exp(-r * t) * norm_cdf(d2)
        price_put = k * cp.exp(-r * t) * norm_cdf(-d2) - s * norm_cdf(-d1)
        return cp.where(is_call, price_call, price_put)
    else:
        # Vectorized for mixed calls/puts using is_call as 1.0 or -1.0 potentially
        # But standard bool where is cleaner
        return cp.zeros_like(s)

def gpu_greeks(
    s: cp.ndarray, k: cp.ndarray, t: cp.ndarray, v: cp.ndarray, r: cp.ndarray, is_call: cp.ndarray
):
    """
    Vectorized Greeks calculation on GPU.
    """
    sqrt_t = cp.sqrt(t)
    d1 = (cp.log(s / k) + (r + 0.5 * v * v) * t) / (v * sqrt_t)

    def norm_cdf(x):
        return 0.5 * (1.0 + cp.erf(x / cp.sqrt(2.0)))

    def norm_pdf(x):
        return cp.exp(-0.5 * x * x) / cp.sqrt(2.0 * cp.pi)

    pdf_d1 = norm_pdf(d1)

    delta = cp.where(is_call, norm_cdf(d1), norm_cdf(d1) - 1.0)
    gamma = pdf_d1 / (s * v * sqrt_t)
    vega = s * sqrt_t * pdf_d1 * 0.01

    return delta, gamma, vega
