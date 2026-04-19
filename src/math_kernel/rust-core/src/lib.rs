use pyo3::prelude::*;

/// Black-Scholes analytical solver.
///
/// Chosen for the hot-path specifically because the closed-form solution 
/// offers O(1) complexity compared to O(N) finite-difference methods, 
/// which is critical for real-time risk sweeps across 1M+ positions.
#[pyfunction]
fn bs_call(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return (s - k * (-r * t).exp()).max(0.0);
    }
    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    s * norm_cdf(d1) - k * (-r * t).exp() * norm_cdf(d2)
}

/// Black-Scholes analytical solver (Put).
#[pyfunction]
fn bs_put(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return (k * (-r * t).exp() - s).max(0.0);
    }
    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    k * (-r * t).exp() * norm_cdf(-d2) - s * norm_cdf(-d1)
}

/// Cumulative Normal Distribution approximation.
///
/// We use the Abramowitz & Stegun (1964) approximation (formula 26.2.17) 
/// because it guarantees an error of less than 7.5e-8, which satisfies 
/// the risk management threshold for BSOPT without the overhead of 
/// full numerical integration.
fn norm_cdf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    0.5 * (1.0 + sign * y)
}

/// Python module definition.
#[pymodule]
fn bsopt_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bs_call, m)?)?;
    m.add_function(wrap_pyfunction!(bs_put, m)?)?;
    Ok(())
}
