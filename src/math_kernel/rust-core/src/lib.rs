use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyArrayMethods, Element, PyArray, PyArrayDescrMethods};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Bound;
use pyo3::ffi::Py_XINCREF;
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};
use std::sync::Arc;
use memmap2::Mmap;
use std::fs::File;

const INV_365: f64 = 1.0 / 365.0;

/// High-Precision Normal Distribution Helper
fn get_norm() -> Normal {
    Normal::new(0.0, 1.0).unwrap()
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
        return Ok(if is_call { (s - k).max(0.0) } else { (k - s).max(0.0) });
    }
    let norm = get_norm();
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * v * v) * t) / (v * sqrt_t);
    let d2 = d1 - v * sqrt_t;

    let price = if is_call {
        s * (-q * t).exp() * norm.cdf(d1) - k * (-r * t).exp() * norm.cdf(d2)
    } else {
        k * (-r * t).exp() * norm.cdf(-d2) - s * (-q * t).exp() * norm.cdf(-d1)
    };
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
    let s = s_arr.as_array();
    let k = k_arr.as_array();
    let t = t_arr.as_array();
    let v = v_arr.as_array();
    let r = r_arr.as_array();
    let q = q_arr.as_array();
    let is_call = is_call_arr.as_array();

    let n = s.len();
    let res = unsafe { PyArray1::<f64>::new_bound(py, [n], false) };
    let res_slice = unsafe { res.as_slice_mut().unwrap() };
    
    let norm = Arc::new(get_norm());

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
                si * (-qi * ti).exp() * norm.cdf(d1) - ki * (-ri * ti).exp() * norm.cdf(d2)
            } else {
                ki * (-ri * ti).exp() * norm.cdf(-d2) - si * (-qi * ti).exp() * norm.cdf(-d1)
            };
        }
    });

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

    let norm = get_norm();
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * v * v) * t) / (v * sqrt_t);
    let d2 = d1 - v * sqrt_t;

    let nd1 = ( -0.5 * d1 * d1 ).exp() * 0.3989422804014327;
    let cdf_d1 = norm.cdf(d1);

    let exp_qt = (-q * t).exp();
    let exp_rt = (-r * t).exp();

    let delta = if is_call { exp_qt * cdf_d1 } else { exp_qt * (cdf_d1 - 1.0) };
    let gamma = exp_qt * nd1 / (s * v * sqrt_t);
    let vega = s * exp_qt * nd1 * sqrt_t * 0.01;

    let theta_call = (-(s * v * exp_qt * nd1) / (2.0 * sqrt_t)) + (q * s * exp_qt * cdf_d1)
        - (r * k * exp_rt * norm.cdf(d2));

    let theta = if is_call { theta_call * INV_365 } else { (theta_call + r * k * exp_rt - q * s * exp_qt) * INV_365 };
    let rho = if is_call { k * t * exp_rt * norm.cdf(d2) * 0.01 } else { -k * t * exp_rt * norm.cdf(-d2) * 0.01 };

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
    let delta = unsafe { PyArray1::<f64>::new_bound(py, [n], false) };
    let gamma = unsafe { PyArray1::<f64>::new_bound(py, [n], false) };
    let theta = unsafe { PyArray1::<f64>::new_bound(py, [n], false) };
    let vega = unsafe { PyArray1::<f64>::new_bound(py, [n], false) };
    let rho = unsafe { PyArray1::<f64>::new_bound(py, [n], false) };

    let d_s = unsafe { delta.as_slice_mut().unwrap() };
    let g_s = unsafe { gamma.as_slice_mut().unwrap() };
    let th_s = unsafe { theta.as_slice_mut().unwrap() };
    let v_s = unsafe { vega.as_slice_mut().unwrap() };
    let r_s = unsafe { rho.as_slice_mut().unwrap() };

    let norm = Arc::new(get_norm());

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
                let cdf_d1 = norm.cdf(d1);
                let exp_qt = (-qi * ti).exp();
                let exp_rt = (-ri * ti).exp();

                *d_out = if call { exp_qt * cdf_d1 } else { exp_qt * (cdf_d1 - 1.0) };
                *g_out = exp_qt * nd1 / (si * vi * sqrt_t);
                *v_out = si * exp_qt * nd1 * sqrt_t * 0.01;
                let theta_call = (-(si * vi * exp_qt * nd1) / (2.0 * sqrt_t)) + (qi * si * exp_qt * cdf_d1) - (ri * ki * exp_rt * norm.cdf(d2));
                *th_out = if call { theta_call * INV_365 } else { (theta_call + ri * ki * exp_rt - qi * si * exp_qt) * INV_365 };
                *rh_out = if call { ki * ti * exp_rt * norm.cdf(d2) * 0.01 } else { -ki * ti * exp_rt * norm.cdf(-d2) * 0.01 };
            }
        });

    Ok((delta, gamma, theta, vega, rho))
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
    let res = unsafe { PyArray1::<f64>::new_bound(py, [n], false) };
    let res_slice = unsafe { res.as_slice_mut().unwrap() };

    res_slice.par_iter_mut().enumerate().for_each(|(i, out)| {
        *out = heston_engine::price_heston(s[i], k[i], t[i], r[i], kappa[i], theta[i], sigma[i], rho[i], v0[i]);
    });

    Ok(res)
}

mod heston_engine {
    use std::f64::consts::PI;
    use num_complex::Complex;

    pub fn price_heston(s: f64, k: f64, t: f64, r: f64, kappa: f64, theta: f64, sigma: f64, rho: f64, v0: f64) -> f64 {
        let p1 = 0.5 + (1.0 / PI) * integral(s, k, t, r, kappa, theta, sigma, rho, v0, 1);
        let p2 = 0.5 + (1.0 / PI) * integral(s, k, t, r, kappa, theta, sigma, rho, v0, 2);
        s * p1 - k * (-r * t).exp() * p2
    }

    fn integral(s: f64, k: f64, t: f64, r: f64, kappa: f64, theta: f64, sigma: f64, rho: f64, v0: f64, j: i32) -> f64 {
        let mut sum = 0.0;
        let n = 100;
        let upper_limit = 100.0;
        let dw = upper_limit / n as f64;
        
        for i in 0..n {
            let w = (i as f64 + 0.5) * dw;
            let cf = char_func(s, t, r, kappa, theta, sigma, rho, v0, w, j);
            let val = (Complex::new(0.0, -w * k.ln()).exp() * cf / Complex::new(0.0, w)).re;
            sum += val * dw;
        }
        sum
    }

    fn char_func(s: f64, t: f64, r: f64, kappa: f64, theta: f64, sigma: f64, rho: f64, v0: f64, w: f64, j: i32) -> Complex<f64> {
        let u = if j == 1 { 0.5 } else { -0.5 };
        let b = if j == 1 { kappa - rho * sigma } else { kappa };
        let a = kappa * theta;
        let i_w = Complex::new(0.0, w);
        
        let d = ((rho * sigma * i_w - b).powi(2) - sigma.powi(2) * (2.0 * u * i_w - w.powi(2))).sqrt();
        let g = (b - rho * sigma * i_w + d) / (b - rho * sigma * i_w - d);
        
        let c = r * i_w * t + (a / sigma.powi(2)) * ((b - rho * sigma * i_w + d) * t - 2.0 * ((1.0 - g * (d * t).exp()) / (1.0 - g)).ln());
        let d_val = ((b - rho * sigma * i_w + d) / sigma.powi(2)) * ((1.0 - (d * t).exp()) / (1.0 - g * (d * t).exp()));
        
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
        let header_size = 8;
        let tick_size = 32;
        if self.mmap.len() < header_size { return Ok(0); }
        Ok((self.mmap.len() - header_size) / tick_size)
    }

    /// Extract prices as a zero-copy NumPy array
    pub fn get_prices<'py>(slf: PyRef<'py, Self>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let n_records = slf.get_n_records()?;
        if n_records == 0 { return Ok(PyArray1::zeros_bound(slf.py(), [0], false)); }
        // Offset: Header(8) + Timestamp(8) + SymbolID(4) = 20
        let ptr = unsafe { slf.mmap.as_ptr().add(8 + 12) };
        unsafe { create_strided_array::<f64, numpy::ndarray::Ix1>(slf, ptr, &[n_records as isize], &[32 as isize]) }
    }

    /// Extract volumes as a zero-copy NumPy array
    pub fn get_volumes<'py>(slf: PyRef<'py, Self>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let n_records = slf.get_n_records()?;
        if n_records == 0 { return Ok(PyArray1::zeros_bound(slf.py(), [0], false)); }
        // Offset: Header(8) + Timestamp(8) + SymbolID(4) + Price(8) = 28
        let ptr = unsafe { slf.mmap.as_ptr().add(8 + 20) };
        unsafe { create_strided_array::<f64, numpy::ndarray::Ix1>(slf, ptr, &[n_records as isize], &[32 as isize]) }
    }

    /// Bulk parse ticks into a vector of structs (heavier parsing)
    pub fn parse_all(&self) -> PyResult<Vec<TickData>> {
        let n = self.get_n_records()?;
        let mut ticks = Vec::with_capacity(n);
        let data = &self.mmap[8..];

        for i in 0..n {
            let offset = i * 32;
            let tick_slice = &data[offset..offset + 32];
            
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

unsafe fn create_strided_array<'py, T, D>(
    slf: PyRef<'py, TickDataBuffer>,
    data_ptr: *const u8,
    dims: &[isize],
    strides: &[isize],
) -> PyResult<Bound<'py, PyArray<T, D>>>
where
    T: Element,
    D: numpy::ndarray::Dimension,
{
    let py = slf.py();
    let type_num = T::get_dtype_bound(py).num();
    extern "C" {
        fn PyArray_New(subtype: *mut std::ffi::c_void, nd: i32, dims: *mut isize, type_num: i32, strides: *mut isize, data: *mut std::ffi::c_void, itemsize: i32, flags: i32, obj: *mut std::ffi::c_void) -> *mut pyo3::ffi::PyObject;
        static mut PyArray_Type: pyo3::ffi::PyTypeObject;
    }
    let array_ptr = PyArray_New(&mut PyArray_Type as *mut _ as *mut std::ffi::c_void, dims.len() as i32, dims.as_ptr() as *mut isize, type_num, strides.as_ptr() as *mut isize, data_ptr as *mut std::ffi::c_void, 0, 0x0400, std::ptr::null_mut());
    if array_ptr.is_null() { return Err(PyErr::fetch(py)); }
    let array_bound = Bound::from_owned_ptr(py, array_ptr).downcast_into_unchecked::<PyArray<T, D>>();
    let base_ptr = slf.as_ptr();
    Py_XINCREF(base_ptr);
    #[repr(C)]
    struct PyArrayObject { ob_base: pyo3::ffi::PyObject, data: *mut std::ffi::c_void, nd: i32, dimensions: *mut isize, strides: *mut isize, base: *mut pyo3::ffi::PyObject }
    let array_ffi_ptr = array_ptr as *mut PyArrayObject;
    (*array_ffi_ptr).base = base_ptr;
    Ok(array_bound)
}

#[pyfunction]
fn simulate_gbm_rk4<'py>(
    py: Python<'py>,
    s0_arr: PyReadonlyArray1<f64>,
    mu_arr: PyReadonlyArray1<f64>,
    sigma_arr: PyReadonlyArray1<f64>,
    t: f64,
    dt: f64,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    let s0 = s0_arr.as_array();
    let mu = mu_arr.as_array();
    let sigma = sigma_arr.as_array();

    let n_paths = s0.len();
    let n_steps = (t / dt) as usize;
    let sqrt_dt = dt.sqrt();

    let res = unsafe { PyArray2::<f64>::new_bound(py, [n_steps + 1, n_paths], false) };
    
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

        for i in 1..=n_steps {
            let z = norm.sample(&mut rng);
            let z_sq = z * z;

            let k1 = muj * current_s;
            let k2 = muj * (current_s + 0.5 * dt * k1);
            let k3 = muj * (current_s + 0.5 * dt * k2);
            let k4 = muj * (current_s + dt * k3);
            let drift = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

            let diffusion = sigmaj * current_s * sqrt_dt * z;
            let milstein_correction = 0.5 * sigmaj * sigmaj * current_s * (z_sq - dt);

            current_s += drift + diffusion + milstein_correction;
            if current_s < 0.0 {
                current_s = 1e-10;
            }

            unsafe {
                *base_ptr.add(i * n_paths + j) = current_s;
            }
        }
    });

    Ok(res)
}

#[pymodule]
fn equaflow_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TickDataBuffer>()?;
    m.add_function(wrap_pyfunction!(black_scholes_price, m)?)?;
    m.add_function(wrap_pyfunction!(batch_black_scholes, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(batch_black_scholes_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(batch_heston_price, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_gbm_rk4, m)?)?;
    Ok(())
}
