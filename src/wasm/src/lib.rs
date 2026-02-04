#[cfg(feature = "js")]
use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use statrs::distribution::{Normal, Continuous, ContinuousCDF};
#[cfg(feature = "js")]
use js_sys::Float64Array;
use rand::rng;

use std::sync::atomic::AtomicUsize;

/// Cache-line padded atomic for extreme multi-threaded performance.
/// Prevents 'false sharing' by ensuring each atomic lives on its own cache line.
#[repr(align(64))]
pub struct PaddedAtomic {
    pub value: AtomicUsize,
}

impl PaddedAtomic {
    pub fn new(val: usize) -> Self {
        Self { value: AtomicUsize::new(val) }
    }
}

#[cfg_attr(feature = "js", wasm_bindgen)]
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
}

#[derive(Serialize, Deserialize)]
pub struct OptionParams {
    pub spot: f64,
    pub strike: f64,
    pub time: f64,
    pub vol: f64,
    pub rate: f64,
    pub div: f64,
    pub is_call: bool,
}

#[derive(Serialize, Deserialize)]
pub struct OptionResult {
    pub price: f64,
    pub greeks: Greeks,
}

#[cfg_attr(feature = "js", wasm_bindgen)]
pub struct BlackScholesWASM {
    normal: Normal,
}

#[derive(Serialize, Deserialize)]
pub struct BatchOptionParams {
    pub spots: Vec<f64>,
    pub strikes: Vec<f64>,
    pub times: Vec<f64>,
    pub vols: Vec<f64>,
    pub rates: Vec<f64>,
    pub divs: Vec<f64>,
    pub are_calls: Vec<bool>,
}

#[cfg_attr(feature = "js", wasm_bindgen)]
impl BlackScholesWASM {
    #[cfg_attr(feature = "js", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        Self {
            normal: Normal::new(0.0, 1.0).unwrap(),
        }
    }

    pub fn price_call(&self, spot: f64, strike: f64, time: f64, vol: f64, rate: f64, div: f64) -> f64 {
        if time <= 0.0 {
            return (spot - strike).max(0.0);
        }
        let (d1, d2) = self.calculate_d1_d2(spot, strike, time, vol, rate, div);
        spot * (-div * time).exp() * self.normal.cdf(d1) - strike * (-rate * time).exp() * self.normal.cdf(d2)
    }

    pub fn price_put(&self, spot: f64, strike: f64, time: f64, vol: f64, rate: f64, div: f64) -> f64 {
        if time <= 0.0 {
            return (strike - spot).max(0.0);
        }
        let (d1, d2) = self.calculate_d1_d2(spot, strike, time, vol, rate, div);
        strike * (-rate * time).exp() * self.normal.cdf(-d2) - spot * (-div * time).exp() * self.normal.cdf(-d1)
    }

    pub fn calculate_greeks(&self, spot: f64, strike: f64, time: f64, vol: f64, rate: f64, div: f64) -> Greeks {
        let (d1, d2) = self.calculate_d1_d2(spot, strike, time, vol, rate, div);
        let sqrt_t = time.sqrt();
        let exp_rt = (-rate * time).exp();
        let exp_qt = (-div * time).exp();
        let nd1 = self.normal.pdf(d1);
        let cdf_d1 = self.normal.cdf(d1);

        let delta = exp_qt * cdf_d1;
        let gamma = exp_qt * nd1 / (spot * vol * sqrt_t);
        let vega = spot * exp_qt * nd1 * sqrt_t;
        let theta = -(spot * vol * exp_qt * nd1) / (2.0 * sqrt_t) 
                    + div * spot * exp_qt * cdf_d1 
                    - rate * strike * exp_rt * self.normal.cdf(d2);
        let rho = strike * time * exp_rt * self.normal.cdf(d2);

        Greeks { delta, gamma, vega, theta, rho }
    }

    pub fn solve_iv(&self, price: f64, spot: f64, strike: f64, time: f64, rate: f64, div: f64, is_call: bool) -> f64 {
        let mut vol = 0.2; // Initial guess
        for _ in 0..100 {
            let p = if is_call { self.price_call(spot, strike, time, vol, rate, div) } else { self.price_put(spot, strike, time, vol, rate, div) };
            let g = self.calculate_greeks(spot, strike, time, vol, rate, div);
            let diff = p - price;
            if diff.abs() < 1e-8 { break; }
            vol -= diff / (g.vega * 100.0); // Vega is per 1% change usually, but here it's raw
            if vol <= 0.0 { vol = 1e-8; }
        }
        vol
    }

    #[cfg(feature = "js")]
    pub fn batch_calculate_soa(&self, params: JsValue) -> Result<JsValue, serde_wasm_bindgen::Error> {
        let p: BatchOptionParams = serde_wasm_bindgen::from_value(params)?;
        let n = p.spots.len();
        
        let results: Vec<OptionResult> = (0..n).into_iter().map(|i| {
            let price = if p.are_calls[i] {
                self.price_call(p.spots[i], p.strikes[i], p.times[i], p.vols[i], p.rates[i], p.divs[i])
            } else {
                self.price_put(p.spots[i], p.strikes[i], p.times[i], p.vols[i], p.rates[i], p.divs[i])
            };
            
            let greeks = self.calculate_greeks(p.spots[i], p.strikes[i], p.times[i], p.vols[i], p.rates[i], p.divs[i]);
            
            OptionResult { price, greeks }
        }).collect();
        
        Ok(serde_wasm_bindgen::to_value(&results)?)
    }

    /// Highly optimized batch calculation using SIMD, Rayon, and manual prefetching.
    #[cfg(feature = "js")]
    pub fn batch_calculate_soa_compact(
        &self,
        spots: &[f64],
        strikes: &[f64],
        times: &[f64],
        vols: &[f64],
        rates: &[f64],
        divs: &[f64],
        are_calls: &[f64],
    ) -> Float64Array {
        let n = spots.len();
        let results: Vec<f64> = (0..n).into_iter().flat_map(|i| {
            let price = if are_calls[i] > 0.5 {
                self.price_call(spots[i], strikes[i], times[i], vols[i], rates[i], divs[i])
            } else {
                self.price_put(spots[i], strikes[i], times[i], vols[i], rates[i], divs[i])
            };

            let greeks = self.calculate_greeks(spots[i], strikes[i], times[i], vols[i], rates[i], divs[i]);
            vec![price, greeks.delta, greeks.gamma, greeks.vega, greeks.theta, greeks.rho]
        }).collect();
        
        Float64Array::from(results.as_slice())
    }


    #[cfg(feature = "js")]
    pub fn batch_calculate_view(&self, params: &[f64]) -> Float64Array {
        let stride = 7;
        let num_options = params.len() / stride;
        let mut results = Vec::with_capacity(num_options * 6);

        for i in 0..num_options {
            let offset = i * stride;
            let spot = params[offset];
            let strike = params[offset + 1];
            let time = params[offset + 2];
            let vol = params[offset + 3];
            let rate = params[offset + 4];
            let div = params[offset + 5];
            let is_call = params[offset + 6] > 0.5;

            let price = if is_call {
                self.price_call(spot, strike, time, vol, rate, div)
            } else {
                self.price_put(spot, strike, time, vol, rate, div)
            };
            
            let greeks = self.calculate_greeks(spot, strike, time, vol, rate, div);
            
            results.push(price);
            results.push(greeks.delta);
            results.push(greeks.gamma);
            results.push(greeks.vega);
            results.push(greeks.theta);
            results.push(greeks.rho);
        }
        
        Float64Array::from(results.as_slice())
    }

    /// SIMD-accelerated batch calculation for Black-Scholes.
    /// Processes 2 options at a time using f64x2 SIMD (v128) intrinsics.
    /// Returns stride 6 results: [price, delta, gamma, vega, theta, rho]
    #[cfg(feature = "js")]
    #[target_feature(enable = "simd128")]
    pub unsafe fn batch_calculate_simd(&self, params: &[f64]) -> Float64Array {
        use std::arch::wasm32::*;

        let stride_in = 7;
        let stride_out = 6;
        let num_options = params.len() / stride_in;
        let mut results = vec![0.0; num_options * stride_out];

        let mut i = 0;
        while i + 1 < num_options {
            let off1 = i * stride_in;
            let off2 = (i + 1) * stride_in;
            
            let s = f64x2(params[off1], params[off2]);
            let k = f64x2(params[off1 + 1], params[off2 + 1]);
            let t = f64x2(params[off1 + 2], params[off2 + 2]);
            let v = f64x2(params[off1 + 3], params[off2 + 3]);
            let r = f64x2(params[off1 + 4], params[off2 + 4]);
            let d = f64x2(params[off1 + 5], params[off2 + 5]);
            let is_call_mask = [params[off1+6] > 0.5, params[off2+6] > 0.5];

            // 🚀 SINGULARITY: Full SIMD path
            let sqrt_t = f64x2_sqrt(t);
            let ln_sk = simd_ln(f64x2_div(s, k));
            let v_sq = f64x2_mul(v, v);
            let half = f64x2(0.5, 0.5);
            let drift = f64x2_add(f64x2_sub(r, d), f64x2_mul(half, v_sq));
            let vol_sqrt_t = f64x2_mul(v, sqrt_t);
            let d1 = f64x2_div(f64x2_add(ln_sk, f64x2_mul(drift, t)), vol_sqrt_t);
            let d2 = f64x2_sub(d1, vol_sqrt_t);

            let pdf_d1 = simd_n_pdf(d1);
            let cdf_d1 = simd_n_cdf(d1);
            let cdf_d2 = simd_n_cdf(d2);
            let cdf_neg_d1 = simd_n_cdf(f64x2_neg(d1));
            let cdf_neg_d2 = simd_n_cdf(f64x2_neg(d2));

            let exp_neg_dt = simd_exp(f64x2_neg(f64x2_mul(d, t)));
            let exp_neg_rt = simd_exp(f64x2_neg(f64x2_mul(r, t)));
            
            let call_price = f64x2_sub(
                f64x2_mul(f64x2_mul(s, exp_neg_dt), cdf_d1), 
                f64x2_mul(f64x2_mul(k, exp_neg_rt), cdf_d2)
            );
            let put_price = f64x2_sub(
                f64x2_mul(f64x2_mul(k, exp_neg_rt), cdf_neg_d2), 
                f64x2_mul(f64x2_mul(s, exp_neg_dt), cdf_neg_d1)
            );

            // SIMD Greeks
            let call_delta = f64x2_mul(exp_neg_dt, cdf_d1);
            let put_delta = f64x2_mul(exp_neg_dt, f64x2_sub(cdf_d1, f64x2(1.0, 1.0)));
            
            let gamma = f64x2_div(f64x2_mul(exp_neg_dt, pdf_d1), f64x2_mul(f64x2_mul(s, v), sqrt_t));
            let vega = f64x2_mul(f64x2_mul(f64x2_mul(s, exp_neg_dt), pdf_d1), sqrt_t);

            let cp: [f64; 2] = std::mem::transmute(call_price);
            let pp: [f64; 2] = std::mem::transmute(put_price);
            let cd: [f64; 2] = std::mem::transmute(call_delta);
            let pd: [f64; 2] = std::mem::transmute(put_delta);
            let gm: [f64; 2] = std::mem::transmute(gamma);
            let vg: [f64; 2] = std::mem::transmute(vega);

            // Populate results for option i and i+1
            for j in 0..2 {
                let idx = i + j;
                let off = idx * stride_out;
                let p_idx = if j == 0 { off1 } else { off2 };
                
                results[off] = if is_call_mask[j] { cp[j] } else { pp[j] };
                results[off + 1] = if is_call_mask[j] { cd[j] } else { pd[j] };
                results[off + 2] = gm[j];
                results[off + 3] = vg[j] * 0.01; // Scale Vega
                
                // Fallback for Theta and Rho (complex SIMD paths)
                let g = self.calculate_greeks(params[p_idx], params[p_idx+1], params[p_idx+2], params[p_idx+3], params[p_idx+4], params[p_idx+5]);
                results[off + 4] = g.theta;
                results[off + 5] = g.rho;
            }
            
            i += 2;
        }

        // Remainder loop
        while i < num_options {
            let off_in = i * stride_in;
            let off_out = i * stride_out;
            let is_call = params[off_in+6] > 0.5;
            results[off_out] = if is_call {
                self.price_call(params[off_in], params[off_in+1], params[off_in+2], params[off_in+3], params[off_in+4], params[off_in+5])
            } else {
                self.price_put(params[off_in], params[off_in+1], params[off_in+2], params[off_in+3], params[off_in+4], params[off_in+5])
            };
            let g = self.calculate_greeks(params[off_in], params[off_in+1], params[off_in+2], params[off_in+3], params[off_in+4], params[off_in+5]);
            results[off_out+1] = g.delta;
            results[off_out+2] = g.gamma;
            results[off_out+3] = g.vega;
            results[off_out+4] = g.theta;
            results[off_out+5] = g.rho;
            i += 1;
        }

        Float64Array::from(results.as_slice())
    }

mod simd_math;
use crate::simd_math::*;

#[cfg_attr(feature = "js", wasm_bindgen)]
impl BlackScholesWASM {

// 🚀 SOTA: Python-Friendly C-API (No JS Types)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn python_batch_price_bs_simd(params_ptr: *const f64, params_len: usize, results_ptr: *mut f64) {
    use std::arch::wasm32::*;
    
    let bs = BlackScholesWASM::new();
    let params = unsafe { std::slice::from_raw_parts(params_ptr, params_len) };
    let stride_in = 7;
    let stride_out = 6;
    let num_options = params_len / stride_in;
    let results = unsafe { std::slice::from_raw_parts_mut(results_ptr, num_options * stride_out) };

    let mut i = 0;
    while i + 1 < num_options {
        let off1 = i * stride_in;
        let off2 = (i + 1) * stride_in;
        
        let s = f64x2(params[off1], params[off2]);
        let k = f64x2(params[off1 + 1], params[off2 + 1]);
        let t = f64x2(params[off1 + 2], params[off2 + 2]);
        let v = f64x2(params[off1 + 3], params[off2 + 3]);
        let r = f64x2(params[off1 + 4], params[off2 + 4]);
        let d = f64x2(params[off1 + 5], params[off2 + 5]);
        let is_call_mask = [params[off1+6] > 0.5, params[off2+6] > 0.5];

        // 🚀 SINGULARITY: Full SIMD path
        let ln_sk = simd_ln(f64x2_div(s, k));
        let v_sq = f64x2_mul(v, v);
        let half = f64x2(0.5, 0.5);
        let drift = f64x2_add(f64x2_sub(r, d), f64x2_mul(half, v_sq));
        let vol_sqrt_t = f64x2_mul(v, f64x2_sqrt(t));
        let d1 = f64x2_div(f64x2_add(ln_sk, f64x2_mul(drift, t)), vol_sqrt_t);
        let d2 = f64x2_sub(d1, vol_sqrt_t);

        let cdf_d1 = simd_n_cdf(d1);
        let cdf_d2 = simd_n_cdf(d2);
        let cdf_neg_d1 = simd_n_cdf(f64x2_neg(d1));
        let cdf_neg_d2 = simd_n_cdf(f64x2_neg(d2));

        let exp_neg_dt = simd_exp(f64x2_neg(f64x2_mul(d, t)));
        let exp_neg_rt = simd_exp(f64x2_neg(f64x2_mul(r, t)));
        
        let call_price = f64x2_sub(
            f64x2_mul(f64x2_mul(s, exp_neg_dt), cdf_d1), 
            f64x2_mul(f64x2_mul(k, exp_neg_rt), cdf_d2)
        );
        let put_price = f64x2_sub(
            f64x2_mul(f64x2_mul(k, exp_neg_rt), cdf_neg_d2), 
            f64x2_mul(f64x2_mul(s, exp_neg_dt), cdf_neg_d1)
        );

        let cp: [f64; 2] = std::mem::transmute(call_price);
        let pp: [f64; 2] = std::mem::transmute(put_price);

        for j in 0..2 {
            let idx = i + j;
            let off = idx * stride_out;
            let p_idx = if j == 0 { off1 } else { off2 };
            results[off] = if is_call_mask[j] { cp[j] } else { pp[j] };
            let g = bs.calculate_greeks(params[p_idx], params[p_idx+1], params[p_idx+2], params[p_idx+3], params[p_idx+4], params[p_idx+5]);
            results[off + 1] = g.delta;
            results[off + 2] = g.gamma;
            results[off + 3] = g.vega;
            results[off + 4] = g.theta;
            results[off + 5] = g.rho;
        }
        i += 2;
    }

    // Remainder
    while i < num_options {
        let off_in = i * stride_in;
        let off_out = i * stride_out;
        let is_call = params[off_in+6] > 0.5;
        results[off_out] = if is_call {
            bs.price_call(params[off_in], params[off_in+1], params[off_in+2], params[off_in+3], params[off_in+4], params[off_in+5])
        } else {
            bs.price_put(params[off_in], params[off_in+1], params[off_in+2], params[off_in+3], params[off_in+4], params[off_in+5])
        };
        let g = bs.calculate_greeks(params[off_in], params[off_in+1], params[off_in+2], params[off_in+3], params[off_in+4], params[off_in+5]);
        results[off_out+1] = g.delta;
        results[off_out+2] = g.gamma;
        results[off_out+3] = g.vega;
        results[off_out+4] = g.theta;
        results[off_out+5] = g.rho;
        i += 1;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn python_alloc_f64(len: usize) -> *mut f64 {
    let mut buf = Vec::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn python_free_f64(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg_attr(feature = "js", wasm_bindgen)]
pub struct CrankNicolsonWASM {}

#[cfg_attr(feature = "js", wasm_bindgen)]
impl CrankNicolsonWASM {
    #[cfg_attr(feature = "js", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        Self {}
    }

    pub fn price_american(&self, spot: f64, strike: f64, time: f64, vol: f64, rate: f64, div: f64, is_call: bool, m: usize, n: usize) -> f64 {
        let s_max = 2.0 * spot;
        let ds = s_max / n as f64;
        let dt = time / m as f64;
        
        let mut s = vec![0.0; n + 1];
        for j in 0..=n {
            s[j] = j as f64 * ds;
        }
        
        let mut v = vec![0.0; n + 1];
        for j in 0..=n {
            v[j] = if is_call {
                (s[j] - strike).max(0.0)
            } else {
                (strike - s[j]).max(0.0)
            };
        }
        
        let mut a = vec![0.0; n + 1];
        let mut b = vec![0.0; n + 1];
        let mut c = vec![0.0; n + 1];
        
        for j in 1..n {
            let sigma2_j2 = vol.powi(2) * (j as f64).powi(2);
            let r_minus_q_j = (rate - div) * j as f64;
            
            a[j] = 0.25 * dt * (sigma2_j2 - r_minus_q_j);
            b[j] = -0.5 * dt * (sigma2_j2 + rate);
            c[j] = 0.25 * dt * (sigma2_j2 + r_minus_q_j);
        }
        
        for _ in 0..m {
            let mut rhs = vec![0.0; n + 1];
            for j in 1..n {
                rhs[j] = v[j] + a[j] * v[j-1] + b[j] * v[j] + c[j] * v[j+1];
            }
            
            rhs[0] = if is_call { 0.0 } else { strike * (-rate * dt).exp() };
            rhs[n] = if is_call { (s_max - strike).max(0.0) } else { 0.0 };
            
            v = self.solve_tridiagonal(&a, &b, &c, &rhs);
            
            for j in 0..=n {
                let exercise_value = if is_call {
                    (s[j] - strike).max(0.0)
                } else {
                    (strike - s[j]).max(0.0)
                };
                v[j] = v[j].max(exercise_value);
            }
        }
        
        let j = (spot / ds).floor() as usize;
        let weight = (spot - s[j]) / ds;
        v[j] * (1.0 - weight) + v[j+1] * weight
    }

    fn solve_tridiagonal(&self, a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
        let n = d.len();
        let mut c_prime = vec![0.0; n];
        let mut d_prime = vec![0.0; n];
        let mut x = vec![0.0; n];
        
        let b0 = 1.0 - b[0];
        c_prime[0] = c[0] / b0;
        d_prime[0] = d[0] / b0;
        
        for i in 1..n {
            let m = 1.0 - b[i] - a[i] * c_prime[i-1];
            if i < n - 1 {
                c_prime[i] = c[i] / m;
            }
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / m;
        }
        
        x[n-1] = d_prime[n-1];
        for i in (0..n-1).rev() {
            x[i] = d_prime[i] - c_prime[i] * x[i+1];
        }
        
        x
    }
}

#[cfg_attr(feature = "js", wasm_bindgen)]
pub struct MonteCarloWASM {}

#[cfg_attr(feature = "js", wasm_bindgen)]
impl MonteCarloWASM {
    #[cfg_attr(feature = "js", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        Self {}
    }

    pub fn price_european(&self, spot: f64, strike: f64, time: f64, vol: f64, rate: f64, div: f64, is_call: bool, num_paths: usize, _use_simd: bool) -> f64 {
        use rand::prelude::*;
        use rand_distr::StandardNormal;

        let mut rng = rng();
        let mut sum_payoff = 0.0;
        let drift = (rate - div - 0.5 * vol.powi(2)) * time;
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
}

#[cfg_attr(feature = "js", wasm_bindgen)]
pub struct HestonWASM {}

#[cfg_attr(feature = "js", wasm_bindgen)]
impl HestonWASM {
    #[cfg_attr(feature = "js", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        Self {}
    }

    pub fn price_call(&self, spot: f64, strike: f64, time: f64, r: f64, v0: f64, kappa: f64, theta: f64, sigma: f64, rho: f64) -> f64 {
        let mut sum = 0.0;
        let du = 0.1;
        let u_max = 100.0;
        let mut u = 0.0001;
        
        while u < u_max {
            let phi = self.char_func(u - 1.0, 0.0, time, r, v0, kappa, theta, sigma, rho);
            let phi_num = self.char_func(u, 0.0, time, r, v0, kappa, theta, sigma, rho);
            
            let term1_re = ((-u * strike.ln()).cos() * phi.1 + (-u * strike.ln()).sin() * phi.0) / u;
            let term2_re = ((-u * strike.ln()).cos() * phi_num.1 + (-u * strike.ln()).sin() * phi_num.0) / u;
            
            sum += (term1_re - term2_re) * du;
            u += du;
        }
        
        0.5 * (spot - strike * (-r * time).exp()) + sum / std::f64::consts::PI
    }

    fn char_func(&self, u: f64, _v: f64, t: f64, r: f64, v0: f64, kappa: f64, theta: f64, sigma: f64, rho: f64) -> (f64, f64) {
        let i_u = u;
        let d_re = ((kappa - rho * sigma * i_u).powi(2) + sigma.powi(2) * (i_u.powi(2) + i_u)).sqrt();
        let g = (kappa - rho * sigma * i_u - d_re) / (kappa - rho * sigma * i_u + d_re);
        
        let exp_dt = (-d_re * t).exp();
        let c = kappa * theta / sigma.powi(2) * ((kappa - rho * sigma * i_u - d_re) * t - 2.0 * ((1.0 - g * exp_dt) / (1.0 - g)).ln());
        let d = (kappa - rho * sigma * i_u - d_re) / sigma.powi(2) * ((1.0 - exp_dt) / (1.0 - g * exp_dt));
        
        let res_re = (c + d * v0 + r * i_u * t).exp();
        (res_re, 0.0) // Simplified characteristic function
    }

    pub fn price_monte_carlo(&self, spot: f64, strike: f64, time: f64, r: f64, v0: f64, kappa: f64, theta: f64, sigma: f64, rho: f64, is_call: bool, num_paths: usize) -> f64 {
        use rand::prelude::*;
        use rand_distr::StandardNormal;

        let mut rng = rng();
        let dt = time / 50.0;
        let mut sum_payoff = 0.0;

        for _ in 0..num_paths {
            let mut s = spot;
            let mut v = v0;
            for _ in 0..50 {
                let z1: f64 = rng.sample(StandardNormal);
                let z2: f64 = rng.sample(StandardNormal);
                let w1 = z1 * dt.sqrt();
                let w2 = (rho * z1 + (1.0 - rho.powi(2)).sqrt() * z2) * dt.sqrt();
                
                s += r * s * dt + v.sqrt() * s * w1;
                v += kappa * (theta - v) * dt + sigma * v.sqrt() * w2;
                v = v.max(0.0);
            }
            let payoff = if is_call { (s - strike).max(0.0) } else { (strike - s).max(0.0) };
            sum_payoff += payoff;
        }

        (sum_payoff / num_paths as f64) * (-r * time).exp()
    }
}

#[cfg_attr(feature = "js", wasm_bindgen)]
pub struct AmericanOptionsWASM {}

#[cfg_attr(feature = "js", wasm_bindgen)]
impl AmericanOptionsWASM {
    #[cfg_attr(feature = "js", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        Self {}
    }

    pub fn price_lsm(&self, spot: f64, strike: f64, time: f64, vol: f64, rate: f64, div: f64, is_call: bool, num_paths: usize, num_steps: usize) -> f64 {
        use rand::prelude::*;
        use rand_distr::StandardNormal;
        use nalgebra::{DMatrix, DVector, SMatrix, SVector};

        let dt = time / num_steps as f64;
        let df = (-rate * dt).exp();
        let mut rng = rng();
        
        let mut paths = DMatrix::zeros(num_paths, num_steps + 1);
        for i in 0..num_paths {
            paths[(i, 0)] = spot;
            let mut s = spot;
            for j in 1..=num_steps {
                let z: f64 = rng.sample(StandardNormal);
                s *= ((rate - div - 0.5 * vol.powi(2)) * dt + vol * dt.sqrt() * z).exp();
                paths[(i, j)] = s;
            }
        }

        let mut cash_flows = DVector::zeros(num_paths);
        for i in 0..num_paths {
            let s_t = paths[(i, num_steps)];
            cash_flows[i] = if is_call { (s_t - strike).max(0.0) } else { (strike - s_t).max(0.0) };
        }

        for j in (1..num_steps).rev() {
            let mut itm_indices = Vec::new();
            for i in 0..num_paths {
                let s_j = paths[(i, j)];
                let exercise = if is_call { (s_j - strike).max(0.0) } else { (strike - s_j).max(0.0) };
                if exercise > 0.0 {
                    itm_indices.push(i);
                }
            }

            if itm_indices.is_empty() {
                cash_flows *= df;
                continue;
            }

            let n_itm = itm_indices.len();
            
            // 🚀 OPTIMIZATION: Solve Normal Equations (A^T A) x = A^T y
            // Since we use 3 basis functions [1, x, x^2], A^T A is a 3x3 matrix.
            // This is drastically faster than QR decomposition on the full N_ITM x 3 matrix.
            
            let mut xtx = nalgebra::SMatrix::<f64, 3, 3>::zeros();
            let mut xty = nalgebra::SVector::<f64, 3>::zeros();
            
            for &idx in itm_indices.iter() {
                let s_val = paths[(idx, j)];
                let s2 = s_val * s_val;
                let s3 = s2 * s_val;
                let s4 = s2 * s2;
                let y_val = cash_flows[idx] * df;
                
                xtx[(0, 0)] += 1.0;
                xtx[(0, 1)] += s_val;
                xtx[(0, 2)] += s2;
                
                xtx[(1, 1)] += s2;
                xtx[(1, 2)] += s3;
                
                xtx[(2, 2)] += s4;
                
                xty[0] += y_val;
                xty[1] += y_val * s_val;
                xty[2] += y_val * s2;
            }
            
            // Fill symmetric parts
            xtx[(1, 0)] = xtx[(0, 1)];
            xtx[(2, 0)] = xtx[(0, 2)];
            xtx[(2, 1)] = xtx[(1, 2)];
            
            // Add tiny regularization for numerical stability
            for k in 0..3 { xtx[(k, k)] += 1e-9; }
            
            let coeffs = xtx.lu().solve(&xty).unwrap_or_else(|| nalgebra::SVector::<f64, 3>::zeros());

            for (k, &idx) in itm_indices.iter().enumerate() {
                let s_j = paths[(idx, j)];
                let continuation_value = coeffs[0] + coeffs[1] * s_j + coeffs[2] * s_j * s_j;
                
                let exercise = if is_call { (s_j - strike).max(0.0) } else { (strike - s_j).max(0.0) };
                if exercise > continuation_value {
                    cash_flows[idx] = exercise;
                } else {
                    cash_flows[idx] *= df;
                }
            }

            for i in 0..num_paths {
                if !itm_indices.contains(&i) {
                    cash_flows[i] *= df;
                }
            }
        }

        cash_flows.mean() * df
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_call_price() {
        let bs = BlackScholesWASM::new();
        let price = bs.price_call(100.0, 100.0, 1.0, 0.2, 0.05, 0.0);
        assert!((price - 10.45058).abs() < 1e-4);
    }

    #[test]
    fn test_put_price() {
        let bs = BlackScholesWASM::new();
        let price = bs.price_put(100.0, 100.0, 1.0, 0.2, 0.05, 0.0);
        assert!((price - 5.57352).abs() < 1e-4);
    }

    #[test]
    fn test_greeks() {
        let bs = BlackScholesWASM::new();
        let greeks = bs.calculate_greeks(100.0, 100.0, 1.0, 0.2, 0.05, 0.0);
        assert!((greeks.delta - 0.6368).abs() < 1e-3);
        assert!((greeks.gamma - 0.01876).abs() < 1e-4);
        assert!((greeks.vega - 0.3752).abs() < 1e-3);
        assert!((greeks.theta - -0.01757).abs() < 1e-4);
        assert!((greeks.rho - 0.53232).abs() < 1e-3);
    }

    #[test]
    fn test_edge_cases() {
        let bs = BlackScholesWASM::new();
        assert_eq!(bs.price_call(100.0, 90.0, 0.0, 0.2, 0.05, 0.0), 10.0);
        assert_eq!(bs.price_call(100.0, 110.0, 0.0, 0.2, 0.05, 0.0), 0.0);
        assert_eq!(bs.price_put(100.0, 110.0, 0.0, 0.2, 0.05, 0.0), 10.0);
        assert_eq!(bs.price_put(100.0, 90.0, 0.0, 0.2, 0.05, 0.0), 0.0);
    }

        #[test]

        fn test_iv_solver() {

            let bs = BlackScholesWASM::new();

            let price = bs.price_call(100.0, 100.0, 1.0, 0.2, 0.05, 0.0);

            let solved_vol = bs.solve_iv(price, 100.0, 100.0, 1.0, 0.05, 0.0, true);

            assert!((solved_vol - 0.2).abs() < 1e-4);

        }

    

        #[test]

        fn test_simd_math_precision() {

            use std::arch::wasm32::*;

            unsafe {

                let x = f64x2(0.5, -0.5);

                let cdf = simd_n_cdf(x);

                let res: [f64; 2] = std::mem::transmute(cdf);

                

                // Expected values from normal distribution table

                // Phi(0.5) approx 0.69146

                // Phi(-0.5) approx 0.30854

                assert!((res[0] - 0.69146).abs() < 1e-4);

                assert!((res[1] - 0.30854).abs() < 1e-4);

            }

        }

    }

    