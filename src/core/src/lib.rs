use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use ndarray::Zip;
use rayon::prelude::*;
use statrs::distribution::{Normal, Continuous, ContinuousCDF};

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

#[pymodule]
fn bsopt_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Greeks>()?;
    m.add_function(wrap_pyfunction!(black_scholes_price, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(batch_black_scholes, m)?)?;
    m.add_function(wrap_pyfunction!(monte_carlo_price, m)?)?;
    Ok(())
}
