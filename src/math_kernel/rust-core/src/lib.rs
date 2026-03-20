use memmap2::Mmap;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use std::fs::File;
use std::sync::Arc;

const INV_SQRT_2PI: f64 = 0.398942280401432677939946059934381868;

fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() * INV_SQRT_2PI
}

/// Normal CDF using A&S 7.1.26 rational approximation (10^-7 precision)
fn norm_cdf(x: f64) -> f64 {
    if x < 0.0 {
        return 1.0 - norm_cdf(-x);
    }
    let p = 0.2316419;
    let a1 = 0.319381530;
    let a2 = -0.356563782;
    let a3 = 1.781477937;
    let a4 = -1.821255978;
    let a5 = 1.330274429;

    let t = 1.0 / (1.0 + p * x);
    let poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    1.0 - norm_pdf(x) * poly
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
    if t <= 0.0 {
        return Ok(if is_call {
            (s - k).max(0.0)
        } else {
            (k - s).max(0.0)
        });
    }
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * v * v) * t) / (v * sqrt_t);
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

    let res: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let si = s[i];
            let ki = k[i];
            let ti = t[i];
            let vi = v[i];
            let ri = r[i];
            let qi = q[i];
            let call = is_call[i];

            if ti <= 0.0 {
                return if call {
                    (si - ki).max(0.0)
                } else {
                    (ki - si).max(0.0)
                };
            }
            let sqrt_t = ti.sqrt();
            let d1 = ((si / ki).ln() + (ri - qi + 0.5 * vi * vi) * ti) / (vi * sqrt_t);
            let d2 = d1 - vi * sqrt_t;

            if call {
                si * (-qi * ti).exp() * norm_cdf(d1) - ki * (-ri * ti).exp() * norm_cdf(d2)
            } else {
                ki * (-ri * ti).exp() * norm_cdf(-d2) - si * (-qi * ti).exp() * norm_cdf(-d1)
            }
        })
        .collect();

    Python::with_gil(|py| Ok(res.into_pyarray(py).to_owned()))
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
        return Ok((
            if is_call { call_delta } else { put_delta },
            0.0,
            0.0,
            0.0,
            0.0,
        ));
    }

    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * v * v) * t) / (v * sqrt_t);
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

    let theta_call = (-(s * v * exp_qt * nd1) / (2.0 * sqrt_t)) + (q * s * exp_qt * cdf_d1)
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
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
)> {
    let s = s_arr.as_array();
    let k = k_arr.as_array();
    let t = t_arr.as_array();
    let v = v_arr.as_array();
    let r = r_arr.as_array();
    let q = q_arr.as_array();
    let is_call = is_call_arr.as_array();

    let n = s.len();

    let results: Vec<(f64, f64, f64, f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
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
                return (if call { cd } else { pd }, 0.0, 0.0, 0.0, 0.0);
            }

            let sqrt_t = ti.sqrt();
            let d1 = ((si / ki).ln() + (ri - qi + 0.5 * vi * vi) * ti) / (vi * sqrt_t);
            let d2 = d1 - vi * sqrt_t;

            let nd1 = norm_pdf(d1);
            let cdf_d1 = norm_cdf(d1);

            let exp_qt = (-qi * ti).exp();
            let exp_rt = (-ri * ti).exp();

            let delta = if call {
                exp_qt * cdf_d1
            } else {
                exp_qt * (cdf_d1 - 1.0)
            };
            let gamma = exp_qt * nd1 / (si * vi * sqrt_t);
            let vega = si * exp_qt * nd1 * sqrt_t * 0.01;

            let theta_call = (-(si * vi * exp_qt * nd1) / (2.0 * sqrt_t))
                + (qi * si * exp_qt * cdf_d1)
                - (ri * ki * exp_rt * norm_cdf(d2));

            let theta = if call {
                theta_call / 365.0
            } else {
                (theta_call + ri * ki * exp_rt - qi * si * exp_qt) / 365.0
            };
            let rho = if call {
                ki * ti * exp_rt * norm_cdf(d2) * 0.01
            } else {
                -ki * ti * exp_rt * norm_cdf(-d2) * 0.01
            };

            (delta, gamma, theta, vega, rho)
        })
        .collect();

    let mut delta_vec = Vec::with_capacity(n);
    let mut gamma_vec = Vec::with_capacity(n);
    let mut theta_vec = Vec::with_capacity(n);
    let mut vega_vec = Vec::with_capacity(n);
    let mut rho_vec = Vec::with_capacity(n);

    for (d, g, th, v, rh) in results {
        delta_vec.push(d);
        gamma_vec.push(g);
        theta_vec.push(th);
        vega_vec.push(v);
        rho_vec.push(rh);
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
fn exact_gbm_path(
    s0: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    sigma: PyReadonlyArray1<f64>,
    t: f64,
    steps: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let s0 = s0.as_array();
    let mu = mu.as_array();
    let sigma = sigma.as_array();
    let n_paths = s0.len();
    let dt = t / steps as f64;
    let sqrt_dt = dt.sqrt();

    let result_flat: Vec<f64> = (0..n_paths)
        .into_par_iter()
        .map(|i| {
            let mut path = Vec::with_capacity(steps + 1);
            let s0_i = s0[i];
            let mu_i = mu[i];
            let sigma_i = sigma[i];

            let mut local_rng = match seed {
                Some(s) => StdRng::seed_from_u64(s + i as u64),
                None => StdRng::from_entropy(),
            };
            let normal = StandardNormal;

            path.push(s0_i);
            let mut current_w = 0.0;
            let drift_const = mu_i - 0.5 * sigma_i * sigma_i;

            for step in 1..=steps {
                let dw: f64 = normal.sample(&mut local_rng);
                current_w += dw * sqrt_dt;
                let time = step as f64 * dt;
                let st = s0_i * (drift_const * time + sigma_i * current_w).exp();
                path.push(st);
            }
            path
        })
        .flatten()
        .collect();

    Python::with_gil(|py| {
        let res = PyArray2::from_shape_vec(
            py,
            (n_paths, steps + 1),
            result_flat,
        )?;
        Ok(res.to_owned())
    })
}

#[pyfunction]
fn validate_tick(timestamp: f64, price: f64, volume: f64) -> PyResult<bool> {
    if timestamp <= 0.0 || price <= 0.0 || volume < 0.0 {
        return Ok(false);
    }
    Ok(true)
}

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
        Ok(Self {
            mmap: Arc::new(mmap),
        })
    }

    pub fn size(&self) -> usize {
        self.mmap.len()
    }

    /// Returns a zero-copy numpy view of the tick data.
    /// Format: 8-byte EQUA header, followed by 32-byte records.
    /// Columns: Symbol (bits), Price (f64), Volume (bits), Timestamp (f64)
    pub fn as_numpy(slf: PyRef<'_, Self>) -> PyResult<Py<PyArray2<f64>>> {
        let py = slf.py();
        let header_size = 8;
        let tick_size = 32;
        
        if slf.mmap.len() < header_size {
            return Err(pyo3::exceptions::PyValueError::new_err("File too small for EQUA header"));
        }

        let data_len = slf.mmap.len() - header_size;
        let n_records = data_len / tick_size;
        
        if n_records == 0 {
            return Ok(unsafe { PyArray2::zeros(py, [0, 4], false).to_owned() });
        }

        let ptr = unsafe { slf.mmap.as_ptr().add(header_size) as *const f64 as *mut f64 };
        
        let array = unsafe {
            PyArray2::from_borrowed_ptr(py, ptr, [n_records, 4])
        };

        // Set base to self to ensure Mmap stays alive
        unsafe {
            let array_ptr = array.as_array_ptr() as *mut numpy::ffi::PyArrayObject;
            let base_obj = slf.to_object(py);
            let base_ptr = base_obj.into_ptr();
            (*array_ptr).base = base_ptr;
        }

        Ok(array.to_owned())
    }
}

#[pymodule]
fn equaflow_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TickDataBuffer>()?;
    m.add_function(wrap_pyfunction!(black_scholes_price, m)?)?;
    m.add_function(wrap_pyfunction!(batch_black_scholes, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(batch_black_scholes_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(exact_gbm_path, m)?)?;
    m.add_function(wrap_pyfunction!(validate_tick, m)?)?;
    Ok(())
}
