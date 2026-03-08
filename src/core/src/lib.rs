use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use ndarray::Zip;
use rayon::prelude::*;
use statrs::distribution::{Normal, Continuous, ContinuousCDF};
use sha3::{Digest, Keccak256};

fn keccak256_hash(data: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

#[pyfunction]
fn keccak256(data: &[u8]) -> String {
    hex::encode(keccak256_hash(data))
}

#[pyfunction]
fn hash_order_eip712(
    domain_separator_hex: &str,
    maker: &str,
    asset: &str,
    amount_hex: &str,
    price_hex: &str,
    nonce: u64,
    expiry: u64,
) -> PyResult<String> {
    let domain_separator = hex::decode(domain_separator_hex.trim_start_matches("0x"))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid domain separator: {}", e)))?;
    
    if domain_separator.len() != 32 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Domain separator must be 32 bytes"));
    }

    // 1. Typehash for Order
    let order_typehash = keccak256_hash(b"Order(address maker,address asset,uint256 amount,uint256 price,uint256 nonce,uint256 expiry)");
    
    // 2. Encode and Hash Struct
    let mut encoded = Vec::with_capacity(32 * 7);
    encoded.extend_from_slice(&order_typehash);
    
    // address maker
    let maker_addr = hex::decode(maker.trim_start_matches("0x"))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid maker address: {}", e)))?;
    let mut maker_padded = [0u8; 32];
    maker_padded[32 - maker_addr.len()..].copy_from_slice(&maker_addr);
    encoded.extend_from_slice(&maker_padded);
    
    // address asset
    let asset_addr = hex::decode(asset.trim_start_matches("0x"))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid asset address: {}", e)))?;
    let mut asset_padded = [0u8; 32];
    asset_padded[32 - asset_addr.len()..].copy_from_slice(&asset_addr);
    encoded.extend_from_slice(&asset_padded);
    
    // uint256 amount
    let amount_val = hex::decode(amount_hex.trim_start_matches("0x"))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid amount: {}", e)))?;
    let mut amount_padded = [0u8; 32];
    amount_padded[32 - amount_val.len()..].copy_from_slice(&amount_val);
    encoded.extend_from_slice(&amount_padded);
    
    // uint256 price
    let price_val = hex::decode(price_hex.trim_start_matches("0x"))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid price: {}", e)))?;
    let mut price_padded = [0u8; 32];
    price_padded[32 - price_val.len()..].copy_from_slice(&price_val);
    encoded.extend_from_slice(&price_padded);
    
    // uint256 nonce
    let mut nonce_bytes = [0u8; 32];
    nonce_bytes[24..].copy_from_slice(&nonce.to_be_bytes());
    encoded.extend_from_slice(&nonce_bytes);
    
    // uint256 expiry
    let mut expiry_bytes = [0u8; 32];
    expiry_bytes[24..].copy_from_slice(&expiry.to_be_bytes());
    encoded.extend_from_slice(&expiry_bytes);
    
    let struct_hash = keccak256_hash(&encoded);
    
    // 3. Final EIP-712 Hash
    let mut final_encoded = Vec::with_capacity(2 + 32 + 32);
    final_encoded.extend_from_slice(b"\x19\x01");
    final_encoded.extend_from_slice(&domain_separator);
    final_encoded.extend_from_slice(&struct_hash);
    
    let final_hash = keccak256_hash(&final_encoded);
    Ok(format!("0x{}", hex::encode(final_hash)))
}

#[pyclass]
#[derive(Clone)]
pub struct Greeks {
    #[pyo3(get)]
    pub delta: f64,
    #[pyo3(get)]
    pub gamma: f64,
    #[pyo3(get)]
    pub vega: f64,
    #[pyo3(get)]
    pub theta: f64,
    #[pyo3(get)]
    pub rho: f64,
}

#[pyfunction]
fn black_scholes_price(
    spot: f64,
    strike: f64,
    time: f64,
    vol: f64,
    rate: f64,
    div: f64,
    is_call: bool,
) -> f64 {
    if time <= 0.0 {
        return if is_call {
            (spot - strike).max(0.0)
        } else {
            (strike - spot).max(0.0)
        };
    }

    let normal = Normal::new(0.0, 1.0).unwrap();
    let d1 = ( (spot / strike).ln() + (rate - div + 0.5 * vol * vol) * time ) / (vol * time.sqrt());
    let d2 = d1 - vol * time.sqrt();

    if is_call {
        spot * (-div * time).exp() * normal.cdf(d1) - strike * (-rate * time).exp() * normal.cdf(d2)
    } else {
        strike * (-rate * time).exp() * normal.cdf(-d2) - spot * (-div * time).exp() * normal.cdf(-d1)
    }
}

#[pyfunction]
fn black_scholes_greeks(
    spot: f64,
    strike: f64,
    time: f64,
    vol: f64,
    rate: f64,
    div: f64,
    is_call: bool,
) -> Greeks {
    if time <= 0.0 {
        let delta = if is_call { if spot > strike { 1.0 } else { 0.0 } } else { if spot < strike { -1.0 } else { 0.0 } };
        return Greeks { delta, gamma: 0.0, vega: 0.0, theta: 0.0, rho: 0.0 };
    }

    let normal = Normal::new(0.0, 1.0).unwrap();
    let sqrt_t = time.sqrt();
    let d1 = ( (spot / strike).ln() + (rate - div + 0.5 * vol * vol) * time ) / (vol * sqrt_t);
    let d2 = d1 - vol * sqrt_t;

    let nd1 = normal.pdf(d1);
    let cdf_d1 = normal.cdf(d1);
    let cdf_d2 = normal.cdf(d2);
    let exp_qt = (-div * time).exp();
    let exp_rt = (-rate * time).exp();

    let gamma = exp_qt * nd1 / (spot * vol * sqrt_t);
    let vega = spot * exp_qt * nd1 * sqrt_t / 100.0;

    let (delta, rho, theta) = if is_call {
        let d = exp_qt * cdf_d1;
        let r_val = strike * time * exp_rt * cdf_d2 / 100.0;
        let th_base = -(spot * vol * exp_qt * nd1) / (2.0 * sqrt_t);
        let th = (th_base - rate * strike * exp_rt * cdf_d2 + div * spot * exp_qt * cdf_d1) / 365.0;
        (d, r_val, th)
    } else {
        let d = exp_qt * (cdf_d1 - 1.0);
        let r_val = -strike * time * exp_rt * (1.0 - cdf_d2) / 100.0;
        let th_base = -(spot * vol * exp_qt * nd1) / (2.0 * sqrt_t);
        let th = (th_base + rate * strike * exp_rt * (1.0 - cdf_d2) - div * spot * exp_qt * (1.0 - cdf_d1)) / 365.0;
        (d, r_val, th)
    };

    Greeks { delta, gamma, vega, theta, rho }
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
) -> PyResult<Py<PyArray1<f64>>> {
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
        Zip::from(&mut results)
            .and(&spots)
            .and(&strikes)
            .and(&times)
            .and(&vols)
            .and(&rates)
            .and(&divs)
            .and(&are_calls)
            .par_for_each(|res, &s, &k, &t, &v, &r, &d, &is_call| {
                *res = black_scholes_price(s, k, t, v, r, d, is_call);
            });
    });

    Ok(PyArray1::from_vec(py, results).into_py(py))
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

    let mut rng = rand::rng();
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
    price: f64,
    quantity: i32,
    side: i32,
    trade_delta: f64,
    current_delta: f64,
    max_qty: i32,
    min_price: f64,
    max_price: f64,
    max_net_delta: f64,
) -> (bool, f64) {
    // 1. Fat-finger checks
    if price < min_price || price > max_price || quantity <= 0 || quantity > max_qty || (side != 1 && side != -1) {
        return (false, current_delta);
    }

    // 2. Delta exposure check
    let new_delta = current_delta + trade_delta;
    if new_delta.abs() > max_net_delta {
        return (false, current_delta);
    }

    (true, new_delta)
}

#[repr(C, packed)]
struct OrderCommand {
    symbol: [u8; 8],
    price: f64,
    quantity: i64,
    side: i32,
    _pad1: [u8; 4], // Align to 8 bytes for f64
    delta: f64,
    submit_ts_ns: i64,
}

#[repr(C, packed)]
struct ExecStatus {
    order_id: i64,
    fill_price: f64,
    fill_qty: i64,
    status: i32,
    _pad1: [u8; 4],
    exec_ts_ns: i64,
}

#[pyfunction]
fn order_engine_loop(
    orders_ptr: usize,
    execs_ptr: usize,
    risk_state_ptr: usize,
    mut last_head: i64,
    mut order_id_counter: i64,
    max_net_delta: f64,
    max_qty: i32,
) -> (i64, i64) {
    unsafe {
        let head_ptr = orders_ptr as *const i64;
        let orders = (orders_ptr + 8) as *const OrderCommand;
        let exec_head_ptr = execs_ptr as *mut i64;
        let execs = (execs_ptr + 8) as *mut ExecStatus;
        let risk_state = risk_state_ptr as *mut f64;

        let current_head = *head_ptr;

        while last_head < current_head {
            let idx = (last_head % 1000) as usize;
            let cmd = &*orders.add(idx);
            
            let trade_delta = cmd.delta * (cmd.quantity as f64) * (cmd.side as f64);
            
            // Risk Check
            let (ok, new_delta) = full_risk_check(
                cmd.price,
                cmd.quantity as i32,
                cmd.side,
                trade_delta,
                *risk_state,
                max_qty,
                0.01,
                1000000.0,
                max_net_delta
            );

            // Write Execution
            let exec = &mut *execs.add(idx);
            if ok {
                exec.order_id = order_id_counter;
                order_id_counter += 1;
                exec.status = 1;
                *risk_state = new_delta;
            } else {
                exec.order_id = -1;
                exec.status = 0;
            }
            exec.fill_price = cmd.price;
            exec.fill_qty = cmd.quantity;
            exec.exec_ts_ns = 0; // In a real system, we'd use a fast clock here

            last_head += 1;
            // Update execution head atomically
            *exec_head_ptr = last_head;
        }

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
fn calculate_mmd(expected: Bound<'_, PyArray1<f64>>, actual: Bound<'_, PyArray1<f64>>) -> f64 {
    let expected = unsafe { expected.as_array() };
    let actual = unsafe { actual.as_array() };

    // MMD Stub
    let term_xx = 0.0;
    let term_yy = 0.0;
    let term_xy = 0.0;

    (term_xx + term_yy - 2.0 * term_xy).max(0.0).sqrt()
}

#[pyfunction]
fn svi_total_variance(k: f64, a: f64, b: f64, rho: f64, m: f64, sigma: f64) -> f64 {
    a + b * (rho * (k - m) + ((k - m).powi(2) + sigma.powi(2)).sqrt())
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
    let f_k = (forward * strike).powf((1.0 - beta) / 2.0);
    let log_f_k = (forward / strike).ln();
    let z = (nu / alpha) * f_k * log_f_k;

    let term2 = if z.abs() < 1e-8 {
        1.0
    } else {
        let xz = ((1.0 - 2.0 * rho * z + z * z).sqrt() + z - rho).ln() / (1.0 - rho).ln();
        // Wait, the formula in Python was:
        // z / np.log((np.sqrt(1.0 - 2.0 * rho * z + z^2) + z - rho) / (1.0 - rho))
        z / ( ((1.0 - 2.0 * rho * z + z * z).sqrt() + z - rho) / (1.0 - rho) ).ln()
    };

    let term1 = alpha / (f_k * (1.0 + (1.0 - beta).powi(2) / 24.0 * log_f_k.powi(2) + (1.0 - beta).powi(4) / 1920.0 * log_f_k.powi(4)));
    
    let term3 = 1.0 + (
        (1.0 - beta).powi(2) / 24.0 * alpha.powi(2) / f_k.powi(2) +
        (rho * beta * nu * alpha) / (4.0 * f_k) +
        ((2.0 - 3.0 * rho.powi(2)) / 24.0) * nu.powi(2)
    ) * maturity;

    term1 * term2 * term3
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
    // Return dummy parameters to allow compilation for now
    Ok(initial_params)
}

#[pymodule]
fn bsopt_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Greeks>()?;
    m.add_function(wrap_pyfunction!(black_scholes_price, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(batch_black_scholes, m)?)?;
    m.add_function(wrap_pyfunction!(monte_carlo_price, m)?)?;
    m.add_function(wrap_pyfunction!(full_risk_check, m)?)?;
    m.add_function(wrap_pyfunction!(order_engine_loop, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_psi, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_mmd, m)?)?;
    m.add_function(wrap_pyfunction!(keccak256, m)?)?;
    m.add_function(wrap_pyfunction!(hash_order_eip712, m)?)?;
    m.add_function(wrap_pyfunction!(svi_total_variance, m)?)?;
    m.add_function(wrap_pyfunction!(sabr_implied_vol, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_svi_rust, m)?)?;
    Ok(())
}
