use pyo3::prelude::*;
use memmap2::Mmap;
use std::fs::File;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use rand::Rng;
use rayon::prelude::*;

fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + statrs::function::erf::erf(x / std::f64::consts::SQRT_2))
}

fn norm_pdf(x: f64) -> f64 {
    (1.0 / std::f64::consts::SQRT_2 / std::f64::consts::PI.sqrt()) * (-0.5 * x * x).exp()
}

#[pyfunction]
fn black_scholes_price(
    s: f64, k: f64, t: f64, v: f64, r: f64, q: f64, is_call: bool
) -> PyResult<f64> {
    if t <= 0.0 {
        return Ok(if is_call { (s - k).max(0.0) } else { (k - s).max(0.0) });
    }
    let sqrt_t = t.sqrt();
    let d1 = ( (s/k).ln() + (r - q + 0.5 * v * v) * t ) / (v * sqrt_t);
    let d2 = d1 - v * sqrt_t;

    let price = if is_call {
        s * (-q * t).exp() * norm_cdf(d1) - k * (-r * t).exp() * norm_cdf(d2)
    } else {
        k * (-r * t).exp() * norm_cdf(-d2) - s * (-q * t).exp() * norm_cdf(-d1)
    };
    Ok(price)
}

#[pyfunction]
fn batch_black_scholes(
    s: PyReadonlyArray1<f64>,
    k: PyReadonlyArray1<f64>,
    t: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
    r: PyReadonlyArray1<f64>,
    q: PyReadonlyArray1<f64>,
    is_call: PyReadonlyArray1<bool>,
) -> PyResult<Py<PyArray1<f64>>> {
    let s = s.as_array();
    let k = k.as_array();
    let t = t.as_array();
    let v = v.as_array();
    let r = r.as_array();
    let q = q.as_array();
    let is_call = is_call.as_array();

    let n = s.len();
    
    let res: Vec<f64> = (0..n).into_par_iter().map(|i| {
        let si = s[i]; let ki = k[i]; let ti = t[i];
        let vi = v[i]; let ri = r[i]; let qi = q[i]; let call = is_call[i];

        if ti <= 0.0 {
            return if call { (si - ki).max(0.0) } else { (ki - si).max(0.0) };
        }
        let sqrt_t = ti.sqrt();
        let d1 = ( (si/ki).ln() + (ri - qi + 0.5 * vi * vi) * ti ) / (vi * sqrt_t);
        let d2 = d1 - vi * sqrt_t;

        if call {
            si * (-qi * ti).exp() * norm_cdf(d1) - ki * (-ri * ti).exp() * norm_cdf(d2)
        } else {
            ki * (-ri * ti).exp() * norm_cdf(-d2) - si * (-qi * ti).exp() * norm_cdf(-d1)
        }
    }).collect();

    Python::with_gil(|py| {
        Ok(res.into_pyarray(py).to_owned())
    })
}

#[pyfunction]
fn black_scholes_greeks(
    s: f64, k: f64, t: f64, v: f64, r: f64, q: f64, is_call: bool
) -> PyResult<(f64, f64, f64, f64, f64)> {
    if t <= 0.0 {
        let call_delta = if s > k { 1.0 } else { 0.0 };
        let put_delta = if s < k { -1.0 } else { 0.0 };
        return Ok((if is_call { call_delta } else { put_delta }, 0.0, 0.0, 0.0, 0.0));
    }
    
    let sqrt_t = t.sqrt();
    let d1 = ( (s/k).ln() + (r - q + 0.5 * v * v) * t ) / (v * sqrt_t);
    let d2 = d1 - v * sqrt_t;

    let nd1 = norm_pdf(d1);
    let cdf_d1 = norm_cdf(d1);
    
    let exp_qt = (-q * t).exp();
    let exp_rt = (-r * t).exp();

    let delta = if is_call {
        exp_qt * cdf_d1
    } else {
        exp_qt * (cdf_d1 - 1.0)
    };

    let gamma = exp_qt * nd1 / (s * v * sqrt_t);
    let vega = s * exp_qt * nd1 * sqrt_t * 0.01;

    let theta_call = (-(s * v * exp_qt * nd1) / (2.0 * sqrt_t)) 
                     + (q * s * exp_qt * cdf_d1) 
                     - (r * k * exp_rt * norm_cdf(d2));
    
    let theta = if is_call {
        theta_call / 365.0
    } else {
        (theta_call + r * k * exp_rt - q * s * exp_qt) / 365.0
    };

    let rho = if is_call {
        k * t * exp_rt * norm_cdf(d2) * 0.01
    } else {
        -k * t * exp_rt * norm_cdf(-d2) * 0.01
    };

    Ok((delta, gamma, theta, vega, rho))
}

#[pyfunction]
fn batch_black_scholes_greeks(
    s_arr: PyReadonlyArray1<f64>,
    k_arr: PyReadonlyArray1<f64>,
    t_arr: PyReadonlyArray1<f64>,
    v_arr: PyReadonlyArray1<f64>,
    r_arr: PyReadonlyArray1<f64>,
    q_arr: PyReadonlyArray1<f64>,
    is_call_arr: PyReadonlyArray1<bool>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let s = s_arr.as_array();
    let k = k_arr.as_array();
    let t = t_arr.as_array();
    let v = v_arr.as_array();
    let r = r_arr.as_array();
    let q = q_arr.as_array();
    let is_call = is_call_arr.as_array();

    let n = s.len();
    
    let mut delta_vec = vec![0.0; n];
    let mut gamma_vec = vec![0.0; n];
    let mut theta_vec = vec![0.0; n];
    let mut vega_vec  = vec![0.0; n];
    let mut rho_vec   = vec![0.0; n];

    (0..n).into_par_iter().map(|i| {
        let si = s[i]; let ki = k[i]; let ti = t[i];
        let vi = v[i]; let ri = r[i]; let qi = q[i]; let call = is_call[i];

        if ti <= 0.0 {
            let cd = if si > ki { 1.0 } else { 0.0 };
            let pd = if si < ki { -1.0 } else { 0.0 };
            return (if call { cd } else { pd }, 0.0, 0.0, 0.0, 0.0);
        }
        
        let sqrt_t = ti.sqrt();
        let d1 = ( (si/ki).ln() + (ri - qi + 0.5 * vi * vi) * ti ) / (vi * sqrt_t);
        let d2 = d1 - vi * sqrt_t;

        let nd1 = norm_pdf(d1);
        let cdf_d1 = norm_cdf(d1);
        
        let exp_qt = (-qi * ti).exp();
        let exp_rt = (-ri * ti).exp();

        let delta = if call { exp_qt * cdf_d1 } else { exp_qt * (cdf_d1 - 1.0) };
        let gamma = exp_qt * nd1 / (si * vi * sqrt_t);
        let vega = si * exp_qt * nd1 * sqrt_t * 0.01;

        let theta_call = (-(si * vi * exp_qt * nd1) / (2.0 * sqrt_t)) 
                         + (qi * si * exp_qt * cdf_d1) 
                         - (ri * ki * exp_rt * norm_cdf(d2));
        
        let theta = if call { theta_call / 365.0 } else { (theta_call + ri * ki * exp_rt - qi * si * exp_qt) / 365.0 };
        let rho = if call { ki * ti * exp_rt * norm_cdf(d2) * 0.01 } else { -ki * ti * exp_rt * norm_cdf(-d2) * 0.01 };

        (delta, gamma, theta, vega, rho)
    }).collect_into_vec(&mut (delta_vec.iter_mut().zip(gamma_vec.iter_mut()).zip(theta_vec.iter_mut()).zip(vega_vec.iter_mut()).zip(rho_vec.iter_mut()).map(|((((a, b), c), d), e)| (*a, *b, *c, *d, *e)).collect::<Vec<_>>())); // Wait, collect_into_vec won't work easily on tuples.

    // Manual unzip
    let results: Vec<_> = (0..n).into_par_iter().map(|i| {
        let si = s[i]; let ki = k[i]; let ti = t[i];
        let vi = v[i]; let ri = r[i]; let qi = q[i]; let call = is_call[i];

        if ti <= 0.0 {
            let cd = if si > ki { 1.0 } else { 0.0 };
            let pd = if si < ki { -1.0 } else { 0.0 };
            return (if call { cd } else { pd }, 0.0, 0.0, 0.0, 0.0);
        }
        
        let sqrt_t = ti.sqrt();
        let d1 = ( (si/ki).ln() + (ri - qi + 0.5 * vi * vi) * ti ) / (vi * sqrt_t);
        let d2 = d1 - vi * sqrt_t;

        let nd1 = norm_pdf(d1);
        let cdf_d1 = norm_cdf(d1);
        
        let exp_qt = (-qi * ti).exp();
        let exp_rt = (-ri * ti).exp();

        let delta = if call { exp_qt * cdf_d1 } else { exp_qt * (cdf_d1 - 1.0) };
        let gamma = exp_qt * nd1 / (si * vi * sqrt_t);
        let vega = si * exp_qt * nd1 * sqrt_t * 0.01;

        let theta_call = (-(si * vi * exp_qt * nd1) / (2.0 * sqrt_t)) 
                         + (qi * si * exp_qt * cdf_d1) 
                         - (ri * ki * exp_rt * norm_cdf(d2));
        
        let theta = if call { theta_call / 365.0 } else { (theta_call + ri * ki * exp_rt - qi * si * exp_qt) / 365.0 };
        let rho = if call { ki * ti * exp_rt * norm_cdf(d2) * 0.01 } else { -ki * ti * exp_rt * norm_cdf(-d2) * 0.01 };

        (delta, gamma, theta, vega, rho)
    }).collect();

    for i in 0..n {
        delta_vec[i] = results[i].0;
        gamma_vec[i] = results[i].1;
        theta_vec[i] = results[i].2;
        vega_vec[i]  = results[i].3;
        rho_vec[i]   = results[i].4;
    }

    Python::with_gil(|py| {
        Ok((
            delta_vec.into_pyarray(py).to_owned(),
            gamma_vec.into_pyarray(py).to_owned(),
            theta_vec.into_pyarray(py).to_owned(),
            vega_vec.into_pyarray(py).to_owned(),
            rho_vec.into_pyarray(py).to_owned(),
        ))
    })
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
    
    let mut s = s0.to_owned();
    
    for _ in 0..steps {
        s.as_slice_mut().unwrap().par_iter_mut().enumerate().for_each(|(i, si)| {
            let mut rng = rand::thread_rng();
            let normal = rand_distr::StandardNormal;

            let k1 = mu[i] * (*si);
            let k2 = mu[i] * ((*si) + 0.5 * dt * k1);
            let k3 = mu[i] * ((*si) + 0.5 * dt * k2);
            let k4 = mu[i] * ((*si) + dt * k3);
            
            let drift = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
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
fn mmap_parse_ticks(path: &str) -> PyResult<Py<PyArray1<f64>>> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    
    let data: &[f64] = unsafe {
        let ptr = mmap.as_ptr() as *const f64;
        let len = mmap.len() / std::mem::size_of::<f64>();
        std::slice::from_raw_parts(ptr, len)
    };
    
    let vec_data = data.to_vec();
    Python::with_gil(|py| {
        Ok(vec_data.into_pyarray(py).to_owned())
    })
}

#[pyfunction]
fn validate_tick(_ticker: &str, price: f64, last_price: f64) -> PyResult<bool> {
    if last_price == 0.0 { Ok(true) } else { Ok(((price - last_price).abs() / last_price) < 0.20) }
}

#[pymodule]
fn equaflow_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(batch_black_scholes, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes_price, m)?)?;
    m.add_function(wrap_pyfunction!(batch_black_scholes_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(mmap_parse_ticks, m)?)?;
    m.add_function(wrap_pyfunction!(runge_kutta_4_vectorized, m)?)?;
    m.add_function(wrap_pyfunction!(validate_tick, m)?)?;
    Ok(())
}
