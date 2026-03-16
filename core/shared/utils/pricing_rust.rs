use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use numpy::{PyReadonlyArray1, PyArray1, IntoPyArray};

// Re-use the optimized pricing kernels from our WASM core
// Note: In a real build, we'd share this via a common crate.
fn black_scholes_price(s: f64, k: f64, t: f64, sigma: f64, r: f64, q: f64, is_call: bool) -> f64 {
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma.powi(2)) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    
    // Fast approximation of Normal CDF (ERF based)
    let nd1 = 0.5 * (1.0 + statrs::function::erf::erf(d1 / 2.0f64.sqrt()));
    let nd2 = 0.5 * (1.0 + statrs::function::erf::erf(d2 / 2.0f64.sqrt()));

    if is_call {
        s * (-q * t).exp() * nd1 - k * (-r * t).exp() * nd2
    } else {
        k * (-r * t).exp() * (1.0 - nd2) - s * (-q * t).exp() * (1.0 - nd1)
    }
}

#[pyfunction]
fn batch_price_rust(
    py: Python,
    spots: PyReadonlyArray1<f64>,
    strikes: PyReadonlyArray1<f64>,
    maturities: PyReadonlyArray1<f64>,
    vols: PyReadonlyArray1<f64>,
    rates: PyReadonlyArray1<f64>,
    divs: PyReadonlyArray1<f64>,
    is_calls: PyReadonlyArray1<bool>,
) -> PyResult<Py<PyArray1<f64>>> {
    let n = spots.len();
    
    // Use Rayon for GIL-free parallel execution
    // py.allow_threads ensures other Python threads can run while we compute in Rust
    let results: Vec<f64> = py.allow_threads(|| {
        (0..n).into_par_iter().map(|i| {
            black_scholes_price(
                *spots.get(i).unwrap(),
                *strikes.get(i).unwrap(),
                *maturities.get(i).unwrap(),
                *vols.get(i).unwrap(),
                *rates.get(i).unwrap(),
                *divs.get(i).unwrap(),
                *is_calls.get(i).unwrap(),
            )
        }).collect()
    });

    Ok(results.into_pyarray(py).to_owned())
}

#[pymodule]
fn bsopt_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(batch_price_rust, m)?)?;
    Ok(())
}
