use memmap2::Mmap;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::sync::Arc;

#[pyclass]
pub struct TickDataBuffer {
    mmap: Arc<Mmap>,
}

#[pymethods]
impl TickDataBuffer {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let file = File::open(path).map_err(|e| PyIOError::new_err(format!("Failed to open file: {}", e)))?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| PyIOError::new_err(format!("Failed to map file: {}", e)))? };
        Ok(Self {
            mmap: Arc::new(mmap),
        })
    }

    pub fn size(&self) -> usize {
        self.mmap.len()
    }

    /// Read raw bytes from the buffer (zero-copy view not directly possible in PyO3 without complicated traits, 
    /// so we return a slice copy for simplicity in this version, but logic is mmap-backed).
    pub fn read_at(&self, offset: usize, len: usize) -> PyResult<Vec<u8>> {
        if offset + len > self.mmap.len() {
            return Err(PyIOError::new_err("Offset out of bounds"));
        }
        Ok(self.mmap[offset..offset + len].to_vec())
    }

    /// Optimized batch parser for fixed-size 32-byte binary ticks
    /// Format: Symbol (8b), Price (8b f64), Volume (8b i64), Timestamp (8b f64)
    pub fn parse_ticks_32b(&self, offset: usize, count: usize) -> PyResult<Vec<(String, f64, i64, f64)>> {
        let tick_size = 32;
        if offset + (count * tick_size) > self.mmap.len() {
            return Err(PyIOError::new_err("Buffer overflow during tick parsing"));
        }

        let mut ticks = Vec::with_capacity(count);
        for i in 0..count {
            let start = offset + (i * tick_size);
            let slice = &self.mmap[start..start + tick_size];
            
            // Symbol (8 bytes)
            let symbol = String::from_utf8_lossy(&slice[0..8]).trim_end_matches('\0').to_string();
            
            // Price (f64)
            let price = f64::from_le_bytes(slice[8..16].try_into().unwrap());
            
            // Volume (i64)
            let volume = i64::from_le_bytes(slice[16..24].try_into().unwrap());
            
            // Timestamp (f64)
            let timestamp = f64::from_le_bytes(slice[24..32].try_into().unwrap());
            
            ticks.push((symbol, price, volume, timestamp));
        }
        Ok(ticks)
    }
}

/// Black-Scholes Vectorized (CPU Parallel)
#[pyfunction]
#[pyo3(name = "black_scholes_batch")]
pub fn black_scholes_batch<'py>(
    py: Python<'py>,
    s: PyReadonlyArray1<f64>,
    k: PyReadonlyArray1<f64>,
    t: PyReadonlyArray1<f64>,
    sigma: PyReadonlyArray1<f64>,
    r: PyReadonlyArray1<f64>,
    is_call: bool,
) -> Bound<'py, PyArray1<f64>> {
    let s = s.as_array();
    let k = k.as_array();
    let t = t.as_array();
    let sigma = sigma.as_array();
    let r = r.as_array();

    let n = s.len();
    let mut results = vec![0.0; n];

    results.par_iter_mut().enumerate().for_each(|(i, val)| {
        let si = s[i];
        let ki = k[i];
        let ti = t[i];
        let sigmai = sigma[i];
        let ri = r[i];

        if ti <= 0.0 {
            *val = if is_call { (si - ki).max(0.0) } else { (ki - si).max(0.0) };
            return;
        }

        let d1 = ( (si / ki).ln() + (ri + 0.5 * sigmai.powi(2)) * ti ) / (sigmai * ti.sqrt());
        let d2 = d1 - sigmai * ti.sqrt();

        let n_d1 = 0.5 * (1.0 + erf(d1 / 2.0f64.sqrt()));
        let n_d2 = 0.5 * (1.0 + erf(d2 / 2.0f64.sqrt()));

        if is_call {
            *val = si * n_d1 - ki * (-ri * ti).exp() * n_d2;
        } else {
            let n_minus_d1 = 0.5 * (1.0 + erf(-d1 / 2.0f64.sqrt()));
            let n_minus_d2 = 0.5 * (1.0 + erf(-d2 / 2.0f64.sqrt()));
            *val = ki * (-ri * ti).exp() * n_minus_d2 - si * n_minus_d1;
        }
    });

    results.into_pyarray_bound(py)
}

/// Runge-Kutta 4 for GBM ODE Step
#[pyfunction]
pub fn rk4_gbm_step<'py>(
    py: Python<'py>,
    s: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    sigma: PyReadonlyArray1<f64>,
    dt: f64,
    dw: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let s = s.as_array();
    let mu = mu.as_array();
    let sigma = sigma.as_array();
    let dw = dw.as_array();

    let n = s.len();
    let mut results = vec![0.0; n];

    results.par_iter_mut().enumerate().for_each(|(i, val)| {
        let si = s[i];
        let mui = mu[i];
        let sigmai = sigma[i];
        let dwi = dw[i];

        // f(s) = mu * s
        let k1 = mui * si;
        let k2 = mui * (si + 0.5 * k1 * dt);
        let k3 = mui * (si + 0.5 * k2 * dt);
        let k4 = mui * (si + k3 * dt);

        let s_new = si + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4) + sigmai * si * dwi;
        *val = s_new.max(0.0);
    });

    results.into_pyarray_bound(py)
}

/// Helper function: Error Function (erf) implementation
/// Approximation from Handbook of Mathematical Functions
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[pymodule]
fn bsopt_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TickDataBuffer>()?;
    m.add_function(wrap_pyfunction!(black_scholes_batch, m)?)?;
    m.add_function(wrap_pyfunction!(rk4_gbm_step, m)?)?;
    Ok(())
}
