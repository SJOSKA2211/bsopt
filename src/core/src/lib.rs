use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyArrayMethods};
use rayon::prelude::*;
use statrs::distribution::{Normal, ContinuousCDF};
use sha3::{Digest, Keccak256};
use pyo3::types::IntoPyDict;

#[pyclass]
#[derive(Clone)]
struct Greeks {
    #[pyo3(get)]
    delta: f64,
    #[pyo3(get)]
    gamma: f64,
    #[pyo3(get)]
    vega: f64,
    #[pyo3(get)]
    theta: f64,
    #[pyo3(get)]
    rho: f64,
}

#[pyfunction]
fn black_scholes_price(
    s: f64,
    k: f64,
    t: f64,
    v: f64,
    r: f64,
    d: f64,
    is_call: bool,
) -> f64 {
    if t <= 0.0 {
        return if is_call { (s - k).max(0.0) } else { (k - s).max(0.0) };
    }

    let d1 = ((s / k).ln() + (r - d + 0.5 * v * v) * t) / (v * t.sqrt());
    let d2 = d1 - v * t.sqrt();
    let n = Normal::new(0.0, 1.0).unwrap();

    if is_call {
        s * (-d * t).exp() * n.cdf(d1) - k * (-r * t).exp() * n.cdf(d2)
    } else {
        k * (-r * t).exp() * n.cdf(-d2) - s * (-d * t).exp() * n.cdf(-d1)
    }
}

#[pyfunction]
fn black_scholes_greeks(
    s: f64,
    k: f64,
    t: f64,
    v: f64,
    r: f64,
    d: f64,
    is_call: bool,
) -> Greeks {
    let t_sqrt = t.sqrt();
    let d1 = ((s / k).ln() + (r - d + 0.5 * v * v) * t) / (v * t_sqrt);
    let d2 = d1 - v * t_sqrt;
    let n = Normal::new(0.0, 1.0).unwrap();
    let pdf_d1 = (-(d1 * d1) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();

    let delta = if is_call {
        (-d * t).exp() * n.cdf(d1)
    } else {
        (-d * t).exp() * (n.cdf(d1) - 1.0)
    };
 
    let theta = if is_call {
        (-(s * v * (-d * t).exp() * pdf_d1) / (2.0 * t_sqrt) 
         - r * k * (-r * t).exp() * n.cdf(d2)
         + d * s * (-d * t).exp() * n.cdf(d1))
    } else {
        (-(s * v * (-d * t).exp() * pdf_d1) / (2.0 * t_sqrt) 
         + r * k * (-r * t).exp() * n.cdf(-d2)
         - d * s * (-d * t).exp() * n.cdf(-d1))
    };
 
    Greeks {
        delta,
        gamma: (-d * t).exp() * pdf_d1 / (s * v * t_sqrt),
        vega: s * (-d * t).exp() * t_sqrt * pdf_d1 * 0.01,
        theta: theta / 365.25,
        rho: if is_call {
            k * t * (-r * t).exp() * n.cdf(d2) * 0.01
        } else {
            -k * t * (-r * t).exp() * n.cdf(-d2) * 0.01
        },
    }
}

#[pyfunction]
fn batch_black_scholes(
    py: Python<'_>,
    spots: PyReadonlyArray1<'_, f64>,
    strikes: PyReadonlyArray1<'_, f64>,
    times: PyReadonlyArray1<'_, f64>,
    vols: PyReadonlyArray1<'_, f64>,
    rates: PyReadonlyArray1<'_, f64>,
    divs: PyReadonlyArray1<'_, f64>,
    are_calls: PyReadonlyArray1<'_, bool>,
) -> PyResult<Py<PyAny>> {
    let spots = spots.as_array();
    let strikes = strikes.as_array();
    let times = times.as_array();
    let vols = vols.as_array();
    let rates = rates.as_array();
    let divs = divs.as_array();
    let are_calls = are_calls.as_array();

    let n = spots.shape()[0];
    let mut results = vec![0.0; n];

    // Release GIL for multi-threaded processing
    py.allow_threads(|| {
        results.par_iter_mut().enumerate().for_each(|(i, res)| {
            *res = black_scholes_price(
                spots[i],
                strikes[i],
                times[i],
                vols[i],
                rates[i],
                divs[i],
                are_calls[i],
            );
        });
    });

    Ok(PyArray1::from_vec(py, results).into_py(py))
}

#[pyfunction]
fn batch_black_scholes_greeks(
    py: Python<'_>,
    spots: PyReadonlyArray1<'_, f64>,
    strikes: PyReadonlyArray1<'_, f64>,
    times: PyReadonlyArray1<'_, f64>,
    vols: PyReadonlyArray1<'_, f64>,
    rates: PyReadonlyArray1<'_, f64>,
    divs: PyReadonlyArray1<'_, f64>,
    are_calls: PyReadonlyArray1<'_, bool>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
)> {
    let spots = spots.as_array();
    let strikes = strikes.as_array();
    let times = times.as_array();
    let vols = vols.as_array();
    let rates = rates.as_array();
    let divs = divs.as_array();
    let are_calls = are_calls.as_array();

    let n = spots.shape()[0];
    let mut deltas = vec![0.0; n];
    let mut gammas = vec![0.0; n];
    let mut thetas = vec![0.0; n];
    let mut vegas = vec![0.0; n];
    let mut rhos = vec![0.0; n];

    py.allow_threads(|| {
        let results: Vec<Greeks> = (0..n)
            .into_par_iter()
            .map(|i| {
                black_scholes_greeks(
                    spots[i],
                    strikes[i],
                    times[i],
                    vols[i],
                    rates[i],
                    divs[i],
                    are_calls[i],
                )
            })
            .collect();

        for i in 0..n {
            deltas[i] = results[i].delta;
            gammas[i] = results[i].gamma;
            thetas[i] = results[i].theta;
            vegas[i] = results[i].vega;
            rhos[i] = results[i].rho;
        }
    });

    Ok((
        PyArray1::from_vec(py, deltas).into_py(py),
        PyArray1::from_vec(py, gammas).into_py(py),
        PyArray1::from_vec(py, thetas).into_py(py),
        PyArray1::from_vec(py, vegas).into_py(py),
        PyArray1::from_vec(py, rhos).into_py(py),
    ))
}

#[pyfunction]
fn monte_carlo_price(
    spot: f64,
    strike: f64,
    time: f64,
    vol: f64,
    rate: f64,
    div: f64,
    is_call: bool,
    num_paths: usize,
) -> f64 {
    use rand::prelude::*;
    use rand_distr::StandardNormal;

    let mut rng = rand::thread_rng();
    let mut sum_payoff = 0.0;
    let drift = (rate - div - 0.5 * vol * vol) * time;
    let vol_sqrt_t = vol * time.sqrt();

    for _ in 0..num_paths {
        let z: f64 = rng.sample(StandardNormal);
        let s_t = spot * (drift + vol_sqrt_t * z).exp();
        let payoff = if is_call {
            (s_t - strike).max(0.0)
        } else {
            (strike - s_t).max(0.0)
        };
        sum_payoff += payoff;
    }

    (sum_payoff / num_paths as f64) * (-rate * time).exp()
}

#[pyfunction]
fn full_risk_check(
    _py: Python<'_>,
    price: f64,
    quantity: i32,
    side: i32,
    d_delta: f64,
    d_gamma: f64,
    d_vega: f64,
    current_delta: f64,
    current_gamma: f64,
    current_vega: f64,
    max_qty: i32,
    min_price: f64,
    max_price: f64,
    max_delta: f64,
    max_gamma: f64,
    max_vega: f64,
) -> PyResult<(bool, f64, f64, f64)> {
    // 1. Silicon-level fat-finger checks
    if price < min_price || price > max_price || quantity <= 0 || quantity > max_qty || (side != 1 && side != -1) {
        return Ok((false, current_delta, current_gamma, current_vega));
    }

    // 2. Incremental Risk Validation
    let new_delta = current_delta + d_delta;
    let new_gamma = current_gamma + d_gamma;
    let new_vega = current_vega + d_vega;

    if new_delta.abs() > max_delta || new_gamma.abs() > max_gamma || new_vega.abs() > max_vega {
        return Ok((false, current_delta, current_gamma, current_vega));
    }

    // 3. Success: Return new state
    Ok((true, new_delta, new_gamma, new_vega))
}

#[pyfunction]
fn order_engine_loop(
    _py: Python<'_>,
    orders_ptr: usize,
    execs_ptr: usize,
    risk_ptr: usize,
    mut last_head: u64,
    mut order_id_counter: u64,
    max_delta: f64,
    max_qty: i32,
) -> (u64, u64) {
    unsafe {
        let orders_head_ptr = orders_ptr as *const u64;
        let execs_head_ptr = execs_ptr as *mut u64;
        let current_head = *orders_head_ptr;

        if last_head >= current_head {
            return (last_head, order_id_counter);
        }

        // Pointers to the data segments (skipping 8-byte head)
        // Order Structure (based on Python OrderBuffer): symbol(8), price(d), quantity(i), side(i), delta(d), gamma(d), vega(d)
        // Total size per order ~ 48 bytes
        let order_data_ptr = (orders_ptr + 8) as *const u8;
        let exec_data_ptr = (execs_ptr + 8) as *mut u8;
        let risk_state_ptr = risk_ptr as *mut f64;

        while last_head < current_head {
            let idx = (last_head % 1000) as usize;
            
            // 1. Read Order (Manual offset calculation for speed)
            // Note: This must strictly match the NumPy structured dtype in shm_mesh.py
            let offset = idx * 48; 
            let entry_ptr = order_data_ptr.add(offset);
            
            let price = *(entry_ptr.add(8) as *const f64);
            let qty = *(entry_ptr.add(16) as *const i32);
            let side = *(entry_ptr.add(20) as *const i32);
            let d_delta = *(entry_ptr.add(24) as *const f64);

            // 2. Risk Check
            let current_portfolio_delta = *risk_state_ptr;
            let trade_delta = d_delta * (qty as f64) * (side as f64);
            let new_delta = current_portfolio_delta + trade_delta;

            let ok = qty > 0 && qty <= max_qty && new_delta.abs() <= max_delta;

            // 3. Write Execution
            // Execution Structure: order_id(q), status(i), fill_price(d), fill_qty(i) ~ 28 bytes
            let exec_offset = idx * 32; // Aligned to 8 bytes for safety
            let out_ptr = exec_data_ptr.add(exec_offset);
            
            if ok {
                *(out_ptr as *mut i64) = order_id_counter as i64;
                *(out_ptr.add(8) as *mut i32) = 1; // Success
                *risk_state_ptr = new_delta; // Commit risk state
                order_id_counter += 1;
            } else {
                *(out_ptr as *mut i64) = -1;
                *(out_ptr.add(8) as *mut i32) = 0; // Reject
            }
            
            *(out_ptr.add(16) as *mut f64) = price;
            *(out_ptr.add(24) as *mut i32) = qty;

            last_head += 1;
        }

        // 4. Update Execution Head
        *execs_head_ptr = last_head;

        (last_head, order_id_counter)
    }
}

#[pyfunction]
fn calculate_psi(expected: Bound<'_, PyArray1<f64>>, actual: Bound<'_, PyArray1<f64>>, bins: Bound<'_, PyArray1<f64>>) -> f64 {
    let expected = unsafe { expected.as_array() };
    let actual = unsafe { actual.as_array() };
    let bins = unsafe { bins.as_array() };

    let n_expected = expected.len() as f64;
    let n_actual = actual.len() as f64;
    let eps = 1e-6;

    let mut psi = 0.0;

    for i in 0..bins.len() - 1 {
        let lower = bins[i];
        let upper = bins[i+1];

        let count_expected = expected.iter().filter(|&&x| x >= lower && x < upper).count() as f64;
        let count_actual = actual.iter().filter(|&&x| x >= lower && x < upper).count() as f64;

        let pct_expected = (count_expected / n_expected) + eps;
        let pct_actual = (count_actual / n_actual) + eps;

        psi += (pct_actual - pct_expected) * (pct_actual / pct_expected).ln();
    }

    psi
}

#[pyfunction]
fn calculate_mmd(x: Bound<'_, PyArray2<f64>>, y: Bound<'_, PyArray2<f64>>, sigma: f64) -> f64 {
    let x = unsafe { x.as_array() };
    let y = unsafe { y.as_array() };

    let n = x.nrows();
    let m = y.nrows();

    if n <= 1 || m <= 1 {
        return 0.0;
    }

    let gamma = 1.0 / (2.0 * sigma.powi(2));

    let mut term_xx = 0.0;
    for i in 0..n {
        for j in 0..n {
            if i == j { continue; }
            let mut dist_sq = 0.0;
            for k in 0..x.ncols() {
                dist_sq += (x[[i, k]] - x[[j, k]]).powi(2);
            }
            term_xx += (-gamma * dist_sq).exp();
        }
    }
    term_xx /= (n * (n - 1)) as f64;

    let mut term_yy = 0.0;
    for i in 0..m {
        for j in 0..m {
            if i == j { continue; }
            let mut dist_sq = 0.0;
            for k in 0..y.ncols() {
                dist_sq += (y[[i, k]] - y[[j, k]]).powi(2);
            }
            term_yy += (-gamma * dist_sq).exp();
        }
    }
    term_yy /= (m * (m - 1)) as f64;

    let mut term_xy = 0.0;
    for i in 0..n {
        for j in 0..m {
            let mut dist_sq = 0.0;
            for k in 0..x.ncols() {
                dist_sq += (x[[i, k]] - y[[j, k]]).powi(2);
            }
            term_xy += (-gamma * dist_sq).exp();
        }
    }
    term_xy /= (n * m) as f64;

    (term_xx + term_yy - 2.0 * term_xy).max(0.0).sqrt()
}

#[pyfunction]
fn svi_total_variance(k: f64, a: f64, b: f64, rho: f64, m: f64, sigma: f64) -> f64 {
    a + b * (rho * (k - m) + ((k - m).powi(2) + sigma.powi(2)).sqrt())
}

#[pyfunction]
fn batch_svi_total_variance(
    py: Python<'_>,
    k: PyReadonlyArray1<'_, f64>,
    a: f64,
    b: f64,
    rho: f64,
    m: f64,
    sigma: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let k = k.as_array();
    let n = k.len();
    let results = PyArray1::new_bound(py, [n], false);
    
    // Use rayon for parallelization if n is large
    if n > 1000 {
        let k_vec: Vec<f64> = k.to_vec();
        let res_vec: Vec<f64> = k_vec.par_iter().map(|&ki| {
            a + b * (rho * (ki - m) + ((ki - m).powi(2) + sigma.powi(2)).sqrt())
        }).collect();
        
        let mut results_view = results.as_array_mut();
        for (i, &val) in res_vec.iter().enumerate() {
            results_view[i] = val;
        }
    } else {
        let mut results_view = results.as_array_mut();
        for i in 0..n {
            let ki = k[i];
            results_view[i] = a + b * (rho * (ki - m) + ((ki - m).powi(2) + sigma.powi(2)).sqrt());
        }
    }
    
    Ok(results.into())
}

#[pyfunction]
fn sabr_implied_vol(
    strike: f64,
    forward: f64,
    maturity: f64,
    alpha: f64,
    beta: f64,
    rho: f64,
    nu: f64,
) -> f64 {
    let f_k = forward * strike;
    let log_f_k = (forward / strike).ln();
    
    if (forward - strike).abs() < 1e-10 {
        let f_beta = forward.powf(1.0 - beta);
        return alpha / f_beta * (1.0 + ((1.0 - beta).powi(2) / 24.0 * alpha.powi(2) / forward.powf(2.0 - 2.0 * beta) + 0.25 * rho * beta * nu * alpha / f_beta + (2.0 - 3.0 * rho.powi(2)) / 24.0 * nu.powi(2)) * maturity);
    }

    let z = nu / alpha * f_k.powf((1.0 - beta) / 2.0) * log_f_k;
    let _xz = (((1.0 - 2.0 * rho * z + z * z).sqrt() + z - rho) / (1.0 - rho)).ln();
    
    let den = f_k.powf((1.0 - beta) / 2.0) * (1.0 + (1.0 - beta).powi(2) / 24.0 * log_f_k.powi(2) + (1.0 - beta).powi(4) / 1920.0 * log_f_k.powi(4));
    
    (alpha / den) * (z / _xz) * (1.0 + ((1.0 - beta).powi(2) / 24.0 * alpha.powi(2) / f_k.powf(1.0 - beta) + 0.25 * rho * beta * nu * alpha / f_k.powf((1.0 - beta) / 2.0) + (2.0 - 3.0 * rho.powi(2)) / 24.0 * nu.powi(2)) * maturity)
}

#[pyfunction]
fn batch_sabr_implied_vol(
    py: Python<'_>,
    strike: PyReadonlyArray1<'_, f64>,
    forward: f64,
    maturity: f64,
    alpha: f64,
    beta: f64,
    rho: f64,
    nu: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let strike = strike.as_array();
    let n = strike.len();
    let results = PyArray1::new_bound(py, [n], false);
    
    let mut results_view = results.as_array_mut();
    for i in 0..n {
        results_view[i] = sabr_implied_vol(strike[i], forward, maturity, alpha, beta, rho, nu);
    }
    
    Ok(results.into())
}

#[pyfunction]
fn keccak256(data: &[u8]) -> Vec<u8> {
    let mut hasher = Keccak256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

#[pyfunction]
fn hash_order_eip712(
    _py: Python<'_>,
    _order_data: &Bound<'_, PyAny>,
) -> PyResult<Vec<u8>> {
    // Simplified EIP-712 hashing for now
    Ok(vec![0u8; 32])
}

#[pyfunction]
fn calibrate_svi_rust(
    py: Python<'_>,
    log_moneyness: PyReadonlyArray1<'_, f64>,
    market_vols: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    maturity: f64,
    initial_params: Vec<f64>,
) -> PyResult<Vec<f64>> {
    use argmin::core::{CostFunction, Gradient, Error, Executor, State};
    use argmin::solver::gradientdescent::SteepestDescent;
    use argmin::solver::linesearch::MoreThuenteLineSearch;

    let k = log_moneyness.as_array();
    let target = market_vols.as_array();
    let w = weights.as_array();

    // SVI Problem Struct
    struct SVIProblem<'a> {
        k: ndarray::ArrayView1<'a, f64>,
        target: ndarray::ArrayView1<'a, f64>,
        weights: ndarray::ArrayView1<'a, f64>,
        maturity: f64,
    }

    impl<'a> CostFunction for SVIProblem<'a> {
        type Param = ndarray::Array1<f64>;
        type Output = f64;

        fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            let a = p[0];
            let b = p[1];
            let rho = p[2];
            let m = p[3];
            let sigma = p[4];

            let mut sum_sq = 0.0;
            for i in 0..self.k.len() {
                let var = a + b * (rho * (self.k[i] - m) + ((self.k[i] - m).powi(2) + sigma.powi(2)).sqrt());
                let vol = (var / self.maturity).max(1e-9).sqrt();
                let diff = (vol - self.target[i]) * self.weights[i];
                sum_sq += diff * diff;
            }
            Ok(sum_sq)
        }
    }

    impl<'a> Gradient for SVIProblem<'a> {
        type Param = ndarray::Array1<f64>;
        type Gradient = ndarray::Array1<f64>;

        fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
            let eps = 1e-7;
            let mut grad = ndarray::Array1::zeros(5);
            let base_cost = self.cost(p)?;
            
            for i in 0..5 {
                let mut p_eps = p.clone();
                p_eps[i] += eps;
                let cost_eps = self.cost(&p_eps)?;
                grad[i] = (cost_eps - base_cost) / eps;
            }
            Ok(grad)
        }
    }

    let problem = SVIProblem {
        k,
        target,
        weights: w,
        maturity,
    };

    let init_param = ndarray::Array1::from_vec(initial_params);
    let linesearch = MoreThuenteLineSearch::new();
    let solver = SteepestDescent::new(linesearch);

    let res = Executor::new(problem, solver)
        .configure(|state: argmin::core::IterState<ndarray::Array1<f64>, ndarray::Array1<f64>, (), (), f64>| 
            state.param(init_param).max_iters(100))
        .run();

    match res {
        Ok(executor) => Ok(executor.state().get_best_param().unwrap().to_vec()),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Calibration failed: {}", e))),
    }
}

#[pymodule]
fn bsopt_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Greeks>()?;
    m.add_function(wrap_pyfunction!(black_scholes_price, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(batch_black_scholes, m)?)?;
    m.add_function(wrap_pyfunction!(batch_black_scholes_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(monte_carlo_price, m)?)?;
    m.add_function(wrap_pyfunction!(full_risk_check, m)?)?;
    m.add_function(wrap_pyfunction!(order_engine_loop, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_psi, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_mmd, m)?)?;
    m.add_function(wrap_pyfunction!(keccak256, m)?)?;
    m.add_function(wrap_pyfunction!(hash_order_eip712, m)?)?;
    m.add_function(wrap_pyfunction!(svi_total_variance, m)?)?;
    m.add_function(wrap_pyfunction!(batch_svi_total_variance, m)?)?;
    m.add_function(wrap_pyfunction!(sabr_implied_vol, m)?)?;
    m.add_function(wrap_pyfunction!(batch_sabr_implied_vol, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_svi_rust, m)?)?;
    Ok(())
}
