use pyo3::prelude::*;
use memmap2::Mmap;
use std::fs::File;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use rand::Rng;
use rayon::prelude::*;
use ndarray::prelude::*;

#[pyfunction]
fn black_scholes_vectorized(
    s: PyReadonlyArray1<f64>,
    k: PyReadonlyArray1<f64>,
    t: PyReadonlyArray1<f64>,
    r: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let s = s.as_array();
    let k = k.as_array();
    let t = t.as_array();
    let r = r.as_array();
    let v = v.as_array();

    let n = s.len();
    
    // Using rayon for parallel iterator performance
    let res: Vec<f64> = (0..n).into_par_iter().map(|i| {
        let sqrt_t = t[i].sqrt();
        let d1 = ( (s[i]/k[i]).ln() + (r[i] + 0.5 * v[i] * v[i]) * t[i] ) / (v[i] * sqrt_t);
        let d2 = d1 - v[i] * sqrt_t;
        
        let call = s[i] * norm_cdf(d1) - k[i] * (-r[i] * t[i]).exp() * norm_cdf(d2);
        call
    }).collect();

    Python::with_gil(|py| {
        Ok(res.into_pyarray(py).to_owned())
    })
}

fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + statrs::function::erf::erf(x / std::f64::consts::SQRT_2))
}

#[pyfunction]
fn runge_kutta_4_vectorized(
    s0: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    sigma: PyReadonlyArray1<f64>,
    _t: f64,
    dt: f64,
    steps: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let s0 = s0.as_array();
    let mu = mu.as_array();
    let sigma = sigma.as_array();
    let n = s0.len();
    
    // Convert to owned array for modification
    let mut s = s0.to_owned();
    
    // Note: RNG within parallel iterators requires careful seeding
    // For simplicity in this kernel, we'll keep the outer loop for steps
    // but parallelize the inner loop across the vector of assets.
    
    for _ in 0..steps {
        // Parallelize across assets
        s.as_slice_mut().unwrap().par_iter_mut().enumerate().for_each(|(i, si)| {
            let mut rng = rand::thread_rng();
            let normal = rand_distr::StandardNormal;

            // RK4 for the drift part: f(s) = mu * s
            let k1 = mu[i] * (*si);
            let k2 = mu[i] * ((*si) + 0.5 * dt * k1);
            let k3 = mu[i] * ((*si) + 0.5 * dt * k2);
            let k4 = mu[i] * ((*si) + dt * k3);
            
            let drift = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
            
            // Stochastic component
            let dw: f64 = rng.sample(normal);
            let diffusion = sigma[i] * (*si) * dw * dt.sqrt();
            
            *si += drift + diffusion;
        });
    }

    Python::with_gil(|py| {
        Ok(s.into_pyarray(py).to_owned())
    })
}

#[pyfunction]
fn mmap_parse_ticks(path: &str) -> PyResult<Vec<f64>> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    
    // Zero-copy parsing of binary f64 ticks from memory-mapped file
    let data: &[f64] = unsafe {
        let ptr = mmap.as_ptr() as *const f64;
        let len = mmap.len() / std::mem::size_of::<f64>();
        std::slice::from_raw_parts(ptr, len)
    };
    
    // Return a subset or the whole vec (this copies into a Vec for Python, 
    // but the read from mmap was zero-copy).
    // To be TRULY zero-copy into Python, we would return a numpy array 
    // pointing at the mmap, but that requires careful lifetime management.
    Ok(data.to_vec())
}

#[pyfunction]
fn validate_tick(_ticker: &str, price: f64, last_price: f64) -> PyResult<bool> {
    if last_price == 0.0 {
        return Ok(true);
    }
    let diff = (price - last_price).abs() / last_price;
    Ok(diff < 0.20)
}

#[pymodule]
fn equaflow_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(black_scholes_vectorized, m)?)?;
    m.add_function(wrap_pyfunction!(mmap_parse_ticks, m)?)?;
    m.add_function(wrap_pyfunction!(runge_kutta_4_vectorized, m)?)?;
    m.add_function(wrap_pyfunction!(validate_tick, m)?)?;
    Ok(())
}

