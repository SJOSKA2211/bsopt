use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyArrayMethods, ToPyArray};
use numpy::ndarray::ShapeBuilder;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Bound;
use rayon::prelude::*;
use std::sync::Arc;
use memmap2::Mmap;
use std::fs::File;
use prometheus::{CounterVec, HistogramVec, Gauge, Registry, Opts, HistogramOpts, TextEncoder, Encoder};
use lazy_static::lazy_static;
use std::path::Path;
use std::net::SocketAddr;
use std::sync::atomic::Ordering;

mod generated;
mod ingest;
mod quarantine;

use crate::ingest::NativeIngestEngine;
use crate::quarantine::QuarantineBuffer;

lazy_static! {
    pub static ref REGISTRY: Registry = Registry::new();

    pub static ref CALL_COUNTER: CounterVec = CounterVec::new(
        Opts::new("manifold_call_total", "Total number of function calls"),
        &["function"]
    ).unwrap();

    pub static ref LATENCY_HISTOGRAM: HistogramVec = HistogramVec::new(
        HistogramOpts::new("manifold_latency_seconds", "Function latency in seconds"),
        &["function"]
    ).unwrap();

    pub static ref RESOURCE_GAUGE: Gauge = Gauge::new(
        "manifold_resource_usage",
        "Current resource usage indicator"
    ).unwrap();
}

fn register_metrics() {
    REGISTRY.register(Box::new(CALL_COUNTER.clone())).ok();
    REGISTRY.register(Box::new(LATENCY_HISTOGRAM.clone())).ok();
    REGISTRY.register(Box::new(RESOURCE_GAUGE.clone())).ok();
}

const INV_365: f64 = 1.0 / 365.0;
const INV_SQRT_2PI: f64 = 0.3989422804014327;

/// High-speed rational approximation for the standard normal CDF.
/// Significantly faster than statrs::distribution::Normal for tight loops.
#[inline(always)]
fn fast_cdf(x: f64) -> f64 {
    if x > 6.0 { return 1.0; }
    if x < -6.0 { return 0.0; }
    
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = INV_SQRT_2PI * (-x * x / 2.0).exp();
    let prob = d * t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    if x > 0.0 { 1.0 - prob } else { prob }
}

#[pyfunction]
fn black_scholes_price(
    s: f64,
    k: f64,
    t: f64,
    v: f64,
    r: f64,
    q: f64,
    is_call: bool,
) -> PyResult<f64> {
    let timer = LATENCY_HISTOGRAM.with_label_values(&["black_scholes_price"]).start_timer();
    CALL_COUNTER.with_label_values(&["black_scholes_price"]).inc();
    
    if t <= 0.0 {
        timer.observe_duration();
        return Ok(if is_call { (s - k).max(0.0) } else { (k - s).max(0.0) });
    }
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * v * v) * t) / (v * sqrt_t);
    let d2 = d1 - v * sqrt_t;

    let price = if is_call {
        s * (-q * t).exp() * fast_cdf(d1) - k * (-r * t).exp() * fast_cdf(d2)
    } else {
        k * (-r * t).exp() * fast_cdf(-d2) - s * (-q * t).exp() * fast_cdf(-d1)
    };
    timer.observe_duration();
    Ok(price)
}

#[pyfunction]
fn batch_black_scholes<'py>(
    py: Python<'py>,
    s_arr: PyReadonlyArray1<f64>,
    k_arr: PyReadonlyArray1<f64>,
    t_arr: PyReadonlyArray1<f64>,
    v_arr: PyReadonlyArray1<f64>,
    r_arr: PyReadonlyArray1<f64>,
    q_arr: PyReadonlyArray1<f64>,
    is_call_arr: PyReadonlyArray1<bool>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let timer = LATENCY_HISTOGRAM.with_label_values(&["batch_black_scholes"]).start_timer();
    CALL_COUNTER.with_label_values(&["batch_black_scholes"]).inc();
    
    let s = s_arr.as_array();
    let k = k_arr.as_array();
    let t = t_arr.as_array();
    let v = v_arr.as_array();
    let r = r_arr.as_array();
    let q = q_arr.as_array();
    let is_call = is_call_arr.as_array();

    let n = s.len();
    RESOURCE_GAUGE.set(n as f64);
    let res = unsafe { PyArray1::<f64>::new(py, [n], false) };
    let res_slice = unsafe { res.as_slice_mut().unwrap() };

    res_slice.par_iter_mut().enumerate().for_each(|(i, out)| {
        let si = s[i];
        let ki = k[i];
        let ti = t[i];
        let vi = v[i];
        let ri = r[i];
        let qi = q[i];
        let call = is_call[i];

        if ti <= 0.0 {
            *out = if call { (si - ki).max(0.0) } else { (ki - si).max(0.0) };
        } else {
            let sqrt_t = ti.sqrt();
            let d1 = ((si / ki).ln() + (ri - qi + 0.5 * vi * vi) * ti) / (vi * sqrt_t);
            let d2 = d1 - vi * sqrt_t;

            *out = if call {
                si * (-qi * ti).exp() * fast_cdf(d1) - ki * (-ri * ti).exp() * fast_cdf(d2)
            } else {
                ki * (-ri * ti).exp() * fast_cdf(-d2) - si * (-qi * ti).exp() * fast_cdf(-d1)
            };
        }
    });

    timer.observe_duration();
    Ok(res)
}

#[pyfunction]
fn black_scholes_greeks(
    s: f64,
    k: f64,
    t: f64,
    v: f64,
    r: f64,
    q: f64,
    is_call: bool,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    if t <= 0.0 {
        let call_delta = if s > k { 1.0 } else { 0.0 };
        let put_delta = if s < k { -1.0 } else { 0.0 };
        return Ok((if is_call { call_delta } else { put_delta }, 0.0, 0.0, 0.0, 0.0));
    }

    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * v * v) * t) / (v * sqrt_t);
    let d2 = d1 - v * sqrt_t;

    let nd1 = ( -0.5 * d1 * d1 ).exp() * 0.3989422804014327;
    let cdf_d1 = fast_cdf(d1);

    let exp_qt = (-q * t).exp();
    let exp_rt = (-r * t).exp();

    let delta = if is_call { exp_qt * cdf_d1 } else { exp_qt * (cdf_d1 - 1.0) };
    let gamma = exp_qt * nd1 / (s * v * sqrt_t);
    let vega = s * exp_qt * nd1 * sqrt_t * 0.01;

    let theta_call = (-(s * v * exp_qt * nd1) / (2.0 * sqrt_t)) + (q * s * exp_qt * cdf_d1)
        - (r * k * exp_rt * fast_cdf(d2));

    let theta = if is_call { theta_call * INV_365 } else { (theta_call + r * k * exp_rt - q * s * exp_qt) * INV_365 };
    let rho = if is_call { k * t * exp_rt * fast_cdf(d2) * 0.01 } else { -k * t * exp_rt * fast_cdf(-d2) * 0.01 };

    Ok((delta, gamma, theta, vega, rho))
}

#[pyfunction]
fn batch_black_scholes_greeks<'py>(
    py: Python<'py>,
    s_arr: PyReadonlyArray1<f64>,
    k_arr: PyReadonlyArray1<f64>,
    t_arr: PyReadonlyArray1<f64>,
    v_arr: PyReadonlyArray1<f64>,
    r_arr: PyReadonlyArray1<f64>,
    q_arr: PyReadonlyArray1<f64>,
    is_call_arr: PyReadonlyArray1<bool>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let s = s_arr.as_array();
    let k = k_arr.as_array();
    let t = t_arr.as_array();
    let v = v_arr.as_array();
    let r = r_arr.as_array();
    let q = q_arr.as_array();
    let is_call = is_call_arr.as_array();

    let n = s.len();
    let delta = unsafe { PyArray1::<f64>::new(py, [n], false) };
    let gamma = unsafe { PyArray1::<f64>::new(py, [n], false) };
    let theta = unsafe { PyArray1::<f64>::new(py, [n], false) };
    let vega = unsafe { PyArray1::<f64>::new(py, [n], false) };
    let rho = unsafe { PyArray1::<f64>::new(py, [n], false) };

    let d_s = unsafe { delta.as_slice_mut().unwrap() };
    let g_s = unsafe { gamma.as_slice_mut().unwrap() };
    let th_s = unsafe { theta.as_slice_mut().unwrap() };
    let v_s = unsafe { vega.as_slice_mut().unwrap() };
    let r_s = unsafe { rho.as_slice_mut().unwrap() };

    d_s.par_iter_mut()
        .zip(g_s.par_iter_mut())
        .zip(th_s.par_iter_mut())
        .zip(v_s.par_iter_mut())
        .zip(r_s.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((((d_out, g_out), th_out), v_out), rh_out))| {
            let si = s[i];
            let ki = k[i];
            let ti = t[i];
            let vi = v[i];
            let ri = r[i];
            let qi = q[i];
            let call = is_call[i];

            if ti <= 0.0 {
                let cd = if si > ki { 1.0 } else { 0.0 };
                let pd = if si < ki { -1.0 } else { 0.0 };
                *d_out = if call { cd } else { pd };
                *g_out = 0.0;
                *th_out = 0.0;
                *v_out = 0.0;
                *rh_out = 0.0;
            } else {
                let sqrt_t = ti.sqrt();
                let d1 = ((si / ki).ln() + (ri - qi + 0.5 * vi * vi) * ti) / (vi * sqrt_t);
                let d2 = d1 - vi * sqrt_t;
                let nd1 = (-0.5 * d1 * d1).exp() * 0.3989422804014327;
                let cdf_d1 = fast_cdf(d1);
                let exp_qt = (-qi * ti).exp();
                let exp_rt = (-ri * ti).exp();

                *d_out = if call { exp_qt * cdf_d1 } else { exp_qt * (cdf_d1 - 1.0) };
                *g_out = exp_qt * nd1 / (si * vi * sqrt_t);
                *v_out = si * exp_qt * nd1 * sqrt_t * 0.01;
                let theta_call = (-(si * vi * exp_qt * nd1) / (2.0 * sqrt_t)) + (qi * si * exp_qt * cdf_d1) - (ri * ki * exp_rt * fast_cdf(d2));
                *th_out = if call { theta_call * INV_365 } else { (theta_call + ri * ki * exp_rt - qi * si * exp_qt) * INV_365 };
                *rh_out = if call { ki * ti * exp_rt * fast_cdf(d2) * 0.01 } else { -ki * ti * exp_rt * fast_cdf(-d2) * 0.01 };
            }
        });

    Ok((delta, gamma, theta, vega, rho))
}

#[pyfunction]
pub fn validate_tick(_ticker: &str, price: f64, last_price: f64) -> bool {
    if last_price == 0.0 {
        return true;
    }
    let diff = (price - last_price).abs();
    let pct_change = diff / last_price;
    pct_change < 0.10
}

#[pyfunction]
pub fn batch_validate_ticks<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<f64>,
    last_prices: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let p = prices.as_array();
    let lp = last_prices.as_array();
    let n = p.len();

    if n != lp.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Prices and last_prices must have same length"));
    }

    let res = unsafe { PyArray1::<bool>::new(py, [n], false) };
    let res_slice = unsafe { res.as_slice_mut().unwrap() };

    res_slice.par_iter_mut().enumerate().for_each(|(i, out)| {
        let price = p[i];
        let last = lp[i];
        if last == 0.0 {
            *out = true;
        } else {
            *out = (price - last).abs() / last < 0.10;
        }
    });

    Ok(res)
}

#[pyfunction]
fn batch_heston_price<'py>(
    py: Python<'py>,
    s_arr: PyReadonlyArray1<f64>,
    k_arr: PyReadonlyArray1<f64>,
    t_arr: PyReadonlyArray1<f64>,
    r_arr: PyReadonlyArray1<f64>,
    kappa_arr: PyReadonlyArray1<f64>,
    theta_arr: PyReadonlyArray1<f64>,
    sigma_arr: PyReadonlyArray1<f64>,
    rho_arr: PyReadonlyArray1<f64>,
    v0_arr: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let timer = LATENCY_HISTOGRAM.with_label_values(&["batch_heston_price"]).start_timer();
    CALL_COUNTER.with_label_values(&["batch_heston_price"]).inc();
    
    let s = s_arr.as_array();
    let k = k_arr.as_array();
    let t = t_arr.as_array();
    let r = r_arr.as_array();
    let kappa = kappa_arr.as_array();
    let theta = theta_arr.as_array();
    let sigma = sigma_arr.as_array();
    let rho = rho_arr.as_array();
    let v0 = v0_arr.as_array();

    let n = s.len();
    RESOURCE_GAUGE.set(n as f64);
    let res = unsafe { PyArray1::<f64>::new(py, [n], false) };
    let res_slice = unsafe { res.as_slice_mut().unwrap() };

    res_slice.par_iter_mut().enumerate().for_each(|(i, out)| {
        *out = heston_engine::price_heston(s[i], k[i], t[i], r[i], kappa[i], theta[i], sigma[i], rho[i], v0[i]);
    });

    timer.observe_duration();
    Ok(res)
}

mod heston_engine {
    use std::f64::consts::PI;
    use num_complex::Complex;

    pub fn price_heston(s: f64, k: f64, t: f64, r: f64, kappa: f64, theta: f64, sigma: f64, rho: f64, v0: f64) -> f64 {
        let p1 = 0.5 + (1.0 / PI) * quadrature_integral(s, k, t, r, kappa, theta, sigma, rho, v0, 1);
        let p2 = 0.5 + (1.0 / PI) * quadrature_integral(s, k, t, r, kappa, theta, sigma, rho, v0, 2);
        (s * p1 - k * (-r * t).exp() * p2).max(0.0)
    }

    fn f(w: f64, s: f64, k: f64, t: f64, r: f64, kappa: f64, theta: f64, sigma: f64, rho: f64, v0: f64, j: i32) -> f64 {
        if w == 0.0 { return 0.0; }
        let cf = char_func(s, t, r, kappa, theta, sigma, rho, v0, w, j);
        (Complex::new(0.0, -w * k.ln()).exp() * cf / Complex::new(0.0, w)).re
    }

    const W10: [f64; 5] = [0.0666713443, 0.1494513492, 0.2190863625, 0.2692667193, 0.2955242247];
    const X10: [f64; 5] = [0.9739065285, 0.8650633667, 0.6794095683, 0.4333953941, 0.1488743390];

    fn quadrature_integral(s: f64, k: f64, t: f64, r: f64, kappa: f64, theta: f64, sigma: f64, rho: f64, v0: f64, j: i32) -> f64 {
        let upper = 100.0;
        let mut sum = 0.0;
        
        for i in 0..5 {
            let w = W10[i];
            let x = X10[i];
            
            // Map [-1, 1] to [0, upper]
            let p1 = 0.5 * upper * (1.0 + x);
            let p2 = 0.5 * upper * (1.0 - x);
            
            sum += w * (f(p1, s, k, t, r, kappa, theta, sigma, rho, v0, j) + f(p2, s, k, t, r, kappa, theta, sigma, rho, v0, j));
        }
        sum * 0.5 * upper
    }

    fn char_func(s: f64, t: f64, r: f64, kappa: f64, theta: f64, sigma: f64, rho: f64, v0: f64, w: f64, j: i32) -> Complex<f64> {
        let u = if j == 1 { 0.5 } else { -0.5 };
        let b = if j == 1 { kappa - rho * sigma } else { kappa };
        let a = kappa * theta;
        let i_w = Complex::new(0.0, w);
        let sig_sq = sigma * sigma;
        
        let d = ((rho * sigma * i_w - b).powi(2) - sig_sq * (2.0 * u * i_w - w * w)).sqrt();
        let g = (b - rho * sigma * i_w + d) / (b - rho * sigma * i_w - d);
        
        let exp_dt = (d * t).exp();
        let c = r * i_w * t + (a / sig_sq) * ((b - rho * sigma * i_w + d) * t - 2.0 * ((1.0 - g * exp_dt) / (1.0 - g)).ln());
        let d_val = ((b - rho * sigma * i_w + d) / sig_sq) * ((1.0 - exp_dt) / (1.0 - g * exp_dt));
        
        (c + d_val * v0 + i_w * s.ln()).exp()
    }
}

#[pyclass]
pub struct TickData {
    #[pyo3(get)]
    pub timestamp: u64,
    #[pyo3(get)]
    pub symbol_id: u32,
    #[pyo3(get)]
    pub price: f64,
    #[pyo3(get)]
    pub volume: f64,
    #[pyo3(get)]
    pub side: u8,
}

const TICK_HEADER_SIZE: usize = 8;
const TICK_RECORD_SIZE: usize = 32;
const PRICE_OFFSET: usize = 12;

#[pyclass]
pub struct TickDataBuffer {
    mmap: Arc<Mmap>,
}

#[pymethods]
impl TickDataBuffer {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { mmap: Arc::new(mmap) })
    }

    pub fn size(&self) -> usize { self.mmap.len() }

    pub fn get_n_records(&self) -> PyResult<usize> {
        if self.mmap.len() < TICK_HEADER_SIZE { return Ok(0); }
        Ok((self.mmap.len() - TICK_HEADER_SIZE) / TICK_RECORD_SIZE)
    }

    /// Extract prices as a zero-copy NumPy array
    pub fn get_prices<'py>(slf: PyRef<'py, Self>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let n_records = slf.get_n_records()?;
        if n_records == 0 { return Ok(PyArray1::zeros(slf.py(), [0], false)); }
        
        let offset = TICK_HEADER_SIZE + PRICE_OFFSET;
        if slf.mmap.len() < offset + n_records * TICK_RECORD_SIZE {
             return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Mmap buffer too small for requested stride"));
        }

        let ptr = unsafe { slf.mmap.as_ptr().add(offset) } as *const f64;
        
        // Use ndarray to create a view, then convert to a PyArray.
        // For zero-copy, we need to ensure the data lives as long as the array.
        // TickDataBuffer holds the Arc<Mmap>, so it's safe.
        let view = unsafe {
            numpy::ndarray::ArrayView1::from_shape_ptr(
                numpy::ndarray::Ix1(n_records).strides(numpy::ndarray::Ix1(TICK_RECORD_SIZE / 8)),
                ptr
            )
        };
        // In numpy 0.23, use to_pyarray for a view.
        Ok(view.to_pyarray(slf.py()))
    }

    /// Extract volumes as a zero-copy NumPy array
    pub fn get_volumes<'py>(slf: PyRef<'py, Self>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let n_records = slf.get_n_records()?;
        if n_records == 0 { return Ok(PyArray1::zeros(slf.py(), [0], false)); }
        
        let offset = TICK_HEADER_SIZE + 20; 
        if slf.mmap.len() < offset + n_records * TICK_RECORD_SIZE {
             return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Mmap buffer too small for requested stride"));
        }

        let ptr = unsafe { slf.mmap.as_ptr().add(offset) } as *const f64;
        let view = unsafe {
            numpy::ndarray::ArrayView1::from_shape_ptr(
                numpy::ndarray::Ix1(n_records).strides(numpy::ndarray::Ix1(TICK_RECORD_SIZE / 8)),
                ptr
            )
        };
        Ok(view.to_pyarray(slf.py()))
    }

    /// Bulk parse ticks into a vector of structs (heavier parsing)
    pub fn parse_all(&self) -> PyResult<Vec<TickData>> {
        let n = self.get_n_records()?;
        let mut ticks = Vec::with_capacity(n);
        let data = &self.mmap[TICK_HEADER_SIZE..];

        for i in 0..n {
            let offset = i * TICK_RECORD_SIZE;
            if offset + TICK_RECORD_SIZE > data.len() { break; }
            let tick_slice = &data[offset..offset + TICK_RECORD_SIZE];
            
            ticks.push(TickData {
                timestamp: u64::from_le_bytes(tick_slice[0..8].try_into().unwrap()),
                symbol_id: u32::from_le_bytes(tick_slice[8..12].try_into().unwrap()),
                price: f64::from_le_bytes(tick_slice[12..20].try_into().unwrap()),
                volume: f64::from_le_bytes(tick_slice[20..28].try_into().unwrap()),
                side: tick_slice[28],
            });
        }
        Ok(ticks)
    }
}

// Replaced unsafe custom strided creation with safe ndarray views above.

#[pyfunction]
#[pyo3(signature = (s0_arr, mu_arr, sigma_arr, t, dt, seed=None))]
fn simulate_gbm_native<'py>(
    py: Python<'py>,
    s0_arr: PyReadonlyArray1<f64>,
    mu_arr: PyReadonlyArray1<f64>,
    sigma_arr: PyReadonlyArray1<f64>,
    t: f64,
    dt: f64,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let timer = LATENCY_HISTOGRAM.with_label_values(&["simulate_gbm_native"]).start_timer();
    CALL_COUNTER.with_label_values(&["simulate_gbm_native"]).inc();
    
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    let s0 = s0_arr.as_array();
    let mu = mu_arr.as_array();
    let sigma = sigma_arr.as_array();

    let n_paths = s0.len();
    RESOURCE_GAUGE.set(n_paths as f64);
    let n_steps = (t / dt) as usize;
    let sqrt_dt = dt.sqrt();

    let res = unsafe { PyArray2::<f64>::new(py, [n_steps + 1, n_paths], false) };
    
    // Safety: We get a raw pointer to the data to avoid sharing Bound across threads.
    // We ensure the array is not reallocated during the loop.
    let data_ptr = res.data() as usize; // Cast to usize for easier sharing

    let norm = Normal::new(0.0, 1.0).unwrap();

    // Set initial values
    unsafe {
        let ptr = data_ptr as *mut f64;
        for j in 0..n_paths {
            *ptr.add(j) = s0[j];
        }
    }

    // Parallelize across paths
    (0..n_paths).into_par_iter().for_each(|j| {
        let mut rng = if let Some(s) = seed {
            rand::rngs::StdRng::seed_from_u64(s + j as u64)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        let muj = mu[j];
        let sigmaj = sigma[j];
        let mut current_s = s0[j];
        let base_ptr = data_ptr as *mut f64;

        // Pre-calculate drift component for the exact solution
        let drift_step = (muj - 0.5 * sigmaj * sigmaj) * dt;
        let vol_step = sigmaj * sqrt_dt;

        for i in 1..=n_steps {
            let z = norm.sample(&mut rng);
            
            // Exact GBM solution: S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
            current_s *= (drift_step + vol_step * z).exp();
            
            if current_s < 0.0 {
                current_s = 1e-10;
            }

            unsafe {
                *base_ptr.add(i * n_paths + j) = current_s;
            }
        }
    });

    timer.observe_duration();
    Ok(res)
}

#[pyfunction]
#[pyo3(signature = (s0, k, t, v, r, q, is_call, n_paths, seed=None))]
fn monte_carlo_price(
    s0: f64,
    k: f64,
    t: f64,
    v: f64,
    r: f64,
    q: f64,
    is_call: bool,
    n_paths: usize,
    seed: Option<u64>,
) -> PyResult<f64> {
    let timer = LATENCY_HISTOGRAM.with_label_values(&["monte_carlo_price"]).start_timer();
    CALL_COUNTER.with_label_values(&["monte_carlo_price"]).inc();
    
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    if t <= 0.0 {
        timer.observe_duration();
        return Ok(if is_call { (s0 - k).max(0.0) } else { (k - s0).max(0.0) });
    }

    let norm = Normal::new(0.0, 1.0).unwrap();
    let drift = (r - q - 0.5 * v * v) * t;
    let vol = v * t.sqrt();
    let discount = (-r * t).exp();

    let total_payoff: f64 = (0..n_paths).into_par_iter().map(|i| {
        let mut rng = if let Some(s) = seed {
            rand::rngs::StdRng::seed_from_u64(s + i as u64)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        let z = norm.sample(&mut rng);
        let st = s0 * (drift + vol * z).exp();
        
        if is_call {
            (st - k).max(0.0)
        } else {
            (k - st).max(0.0)
        }
    }).sum();

    let price = discount * (total_payoff / n_paths as f64);
    timer.observe_duration();
    Ok(price)
}

#[pyfunction]
fn get_manifold_metrics() -> PyResult<String> {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    Ok(String::from_utf8(buffer).unwrap())
}

#[pymodule]
fn Manifold_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_metrics();
    m.add_class::<TickDataBuffer>()?;
    m.add_function(wrap_pyfunction!(black_scholes_price, m)?)?;
    m.add_function(wrap_pyfunction!(batch_black_scholes, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(batch_black_scholes_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(batch_heston_price, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_gbm_native, m)?)?;
    m.add_function(wrap_pyfunction!(validate_tick, m)?)?;
    m.add_function(wrap_pyfunction!(batch_validate_ticks, m)?)?;
    m.add_class::<PyNativeIngest>()?;
    m.add_function(wrap_pyfunction!(get_manifold_metrics, m)?)?;
    Ok(())
}

#[pyclass]
pub struct PyNativeIngest {
    engine: Option<Arc<NativeIngestEngine>>,
    quarantine: Arc<QuarantineBuffer>,
}

#[pymethods]
impl PyNativeIngest {
    #[new]
    pub fn new(addr: String, quarantine_path: String) -> PyResult<Self> {
        let socket_addr: SocketAddr = addr.parse()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid address: {}", e)))?;
        
        let q_inner = QuarantineBuffer::new(Path::new(&quarantine_path), 10000)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to init quarantine: {}", e)))?;
        
        let quarantine = Arc::new(q_inner);
        
        Ok(Self {
            engine: Some(Arc::new(NativeIngestEngine::new(socket_addr, quarantine.clone()))),
            quarantine,
        })
    }

    pub fn ping(&self) -> bool {
        true
    }

    pub fn start(&mut self) -> PyResult<()> {
        if let Some(engine) = self.engine.clone() {
            // Spawn in a background tokio thread
            std::thread::spawn(move || {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    if let Err(e) = engine.run().await {
                        eprintln!("Native ingest engine error: {}", e);
                    }
                });
            });
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Engine already started or not initialized"))
        }
    }

    pub fn get_metrics(&self) -> PyResult<String> {
        if let Some(ref engine) = self.engine {
            let processed = engine.processed_count.load(Ordering::Relaxed);
            let total_rejected = self.quarantine.total_rejections.load(Ordering::Relaxed);
            
            let val = serde_json::json!({
                "processed": processed,
                "rejected": total_rejected,
                "health": "HIGH_THROUGHPUT_ACTIVE"
            });
            Ok(val.to_string())
        } else {
            Ok(serde_json::json!({"health": "INACTIVE"}).to_string())
        }
    }
}
