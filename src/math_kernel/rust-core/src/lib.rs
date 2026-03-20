use memmap2::Mmap;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use std::fs::File;

fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + statrs::function::erf::erf(x / std::f64::consts::SQRT_2))
}

fn norm_pdf(x: f64) -> f64 {
    (1.0 / std::f64::consts::SQRT_2 / std::f64::consts::PI.sqrt()) * (-0.5 * x * x).exp()
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
fn runge_kutta_4_gbm(
    s0: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    sigma: PyReadonlyArray1<f64>,
    _t: f64,
    dt: f64,
    steps: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let s0 = s0.as_array();
    let mu = mu.as_array();
    let sigma = sigma.as_array();
    let n_paths = s0.len();
    let sqrt_dt = dt.sqrt();

    // Parallelize path generation using Rayon
    let result_flat: Vec<f64> = (0..n_paths)
        .into_par_iter()
        .map(|i| {
            let mut path = Vec::with_capacity(steps + 1);
            let mut si = s0[i];
            let mu_i = mu[i];
            let sigma_i = sigma[i];

            let mut local_rng = match seed {
                Some(s) => StdRng::seed_from_u64(s + i as u64),
                None => StdRng::from_entropy(),
            };
            let normal = StandardNormal;

            path.push(si);

            for _ in 0..steps {
                let dw: f64 = normal.sample(&mut local_rng);

                // RK4 Deterministic Drift
                let k1 = mu_i * si;
                let k2 = mu_i * (si + 0.5 * dt * k1);
                let k3 = mu_i * (si + 0.5 * dt * k2);
                let k4 = mu_i * (si + dt * k3);
                let drift = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

                // Stochastic Diffusion (Milstein Correction for Strong Convergence)
                let diffusion = sigma_i * si * dw * sqrt_dt;
                let milstein = 0.5 * sigma_i * sigma_i * si * (dw * dw - dt);

                si = si + drift + diffusion + milstein;
                if si < 0.0 {
                    si = 1e-10;
                }
                path.push(si);
            }
            path
        })
        .flatten()
        .collect();

    Python::with_gil(|py| {
        let res = PyArray2::from_vec2(
            py,
            &result_flat
                .chunks(steps + 1)
                .map(|v| v.to_vec())
                .collect::<Vec<_>>(),
        )?;
        Ok(res.to_owned())
    })
}

#[pyfunction]
fn simulate_gbm_euler(
    s0: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    sigma: PyReadonlyArray1<f64>,
    t: f64,
    dt: f64,
    seed: Option<u64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let s0 = s0.as_array();
    let mu = mu.as_array();
    let sigma = sigma.as_array();
    let n_paths = s0.len();

    let sqrt_dt = dt.sqrt();
    let n_steps = (t / dt) as usize;

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let normal = StandardNormal;

    let mut paths = vec![vec![0.0; n_steps + 1]; n_paths];

    for i in 0..n_paths {
        paths[i][0] = s0[i];
    }

    for step in 0..n_steps {
        for i in 0..n_paths {
            let si = paths[i][step];
            let dw: f64 = normal.sample(&mut rng);

            let drift = mu[i] * si * dt;
            let diffusion = sigma[i] * si * sqrt_dt * dw;

            paths[i][step + 1] = si + drift + diffusion;

            if paths[i][step + 1] < 0.0 {
                paths[i][step + 1] = 0.0001;
            }
        }
    }

    Python::with_gil(|py| Ok(PyArray2::from_vec2(py, &paths)?.to_owned()))
}

#[pyfunction]
fn runge_kutta_4_vectorized(
    s0: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    sigma: PyReadonlyArray1<f64>,
    _t: f64,
    dt: f64,
    steps: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let s0 = s0.as_array();
    let mu = mu.as_array();
    let sigma = sigma.as_array();
    let n = s0.len();

    let sqrt_dt = dt.sqrt();
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let normal = StandardNormal;

    let mut s = s0.to_owned();

    for _ in 0..steps {
        for i in 0..n {
            let si = s[i];
            let dw: f64 = normal.sample(&mut rng);

            let k1 = mu[i] * si;
            let k2 = mu[i] * (si + 0.5 * dt * k1);
            let k3 = mu[i] * (si + 0.5 * dt * k2);
            let k4 = mu[i] * (si + dt * k3);

            let drift = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

            let diffusion = sigma[i] * si * dw * sqrt_dt;

            let milstein = 0.5 * sigma[i] * sigma[i] * si * (dw * dw - dt);

            s[i] = si + drift + diffusion + milstein;

            if s[i] < 0.0 {
                s[i] = 0.0001;
            }
        }
    }

    Python::with_gil(|py| Ok(s.into_pyarray(py).to_owned()))
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
    Python::with_gil(|py| Ok(vec_data.into_pyarray(py).to_owned()))
}

#[pyfunction]
fn mmap_parse_structured_ticks(path: &str) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    // Assumption: Binary format [Timestamp (f64), Price (f64)]
    let record_size = std::mem::size_of::<f64>() * 2;
    let n_records = mmap.len() / record_size;

    let mut timestamps = Vec::with_capacity(n_records);
    let mut prices = Vec::with_capacity(n_records);

    unsafe {
        let ptr = mmap.as_ptr() as *const f64;
        for i in 0..n_records {
            timestamps.push(*ptr.add(i * 2));
            prices.push(*ptr.add(i * 2 + 1));
        }
    }

    Python::with_gil(|py| {
        Ok((
            timestamps.into_pyarray(py).to_owned(),
            prices.into_pyarray(py).to_owned(),
        ))
    })
}

#[pyfunction]
fn validate_tick(timestamp: f64, price: f64, volume: f64) -> PyResult<bool> {
    if timestamp <= 0.0 || price <= 0.0 || volume < 0.0 {
        return Ok(false);
    }
    Ok(true)
}

#[pymodule]
fn equaflow_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(batch_black_scholes, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes_price, m)?)?;
    m.add_function(wrap_pyfunction!(batch_black_scholes_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(mmap_parse_ticks, m)?)?;
    m.add_function(wrap_pyfunction!(mmap_parse_structured_ticks, m)?)?;
    m.add_function(wrap_pyfunction!(runge_kutta_4_vectorized, m)?)?;
    m.add_function(wrap_pyfunction!(runge_kutta_4_gbm, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_gbm_euler, m)?)?;
    m.add_function(wrap_pyfunction!(validate_tick, m)?)?;
    Ok(())
}
