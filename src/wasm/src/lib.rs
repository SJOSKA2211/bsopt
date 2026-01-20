// Verified by Conductor
use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use statrs::distribution::{Normal, Continuous, ContinuousCDF};
use js_sys::Float64Array;

#[wasm_bindgen]
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

#[wasm_bindgen]
pub struct BlackScholesWASM {
    normal: Normal,
}

#[wasm_bindgen]
impl BlackScholesWASM {
    #[wasm_bindgen(constructor)]
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
        let cdf_d2 = self.normal.cdf(d2);

        let delta = exp_qt * cdf_d1;
        let gamma = exp_qt * nd1 / (spot * vol * sqrt_t);
        let vega = spot * exp_qt * nd1 * sqrt_t;
        let theta = -(spot * vol * exp_qt * nd1) / (2.0 * sqrt_t) 
                    - rate * strike * exp_rt * cdf_d2 
                    + div * spot * exp_qt * cdf_d1;
        let rho = strike * time * exp_rt * cdf_d2;

        Greeks {
            delta,
            gamma,
            vega: vega / 100.0, // Per 1% change
            theta: theta / 365.0, // Per day
            rho: rho / 100.0, // Per 1% change
        }
    }

    pub fn solve_iv(&self, price: f64, spot: f64, strike: f64, time: f64, rate: f64, div: f64, is_call: bool) -> f64 {
        let mut vol = 0.5; // Initial guess
        let max_iter = 100;
        let epsilon = 1e-8;

        for _ in 0..max_iter {
            let p = if is_call {
                self.price_call(spot, strike, time, vol, rate, div)
            } else {
                self.price_put(spot, strike, time, vol, rate, div)
            };
            
            let diff = p - price;
            if diff.abs() < epsilon {
                return vol;
            }

            let vega = self.calculate_vega(spot, strike, time, vol, rate, div);
            if vega.abs() < 1e-10 {
                break;
            }

            vol = vol - diff / vega;
            
            if vol <= 0.0 {
                vol = 1e-6; // Ensure vol stays positive
            }
        }
        vol
    }

    fn calculate_vega(&self, spot: f64, strike: f64, time: f64, vol: f64, rate: f64, div: f64) -> f64 {
        let (d1, _) = self.calculate_d1_d2(spot, strike, time, vol, rate, div);
        let exp_qt = (-div * time).exp();
        let nd1 = self.normal.pdf(d1);
        spot * exp_qt * nd1 * time.sqrt()
    }

    pub fn batch_calculate(&self, params: JsValue) -> Result<JsValue, serde_wasm_bindgen::Error> {
        let options: Vec<OptionParams> = serde_wasm_bindgen::from_value(params)?;
        let results = self.batch_calculate_internal(options);
        Ok(serde_wasm_bindgen::to_value(&results)?)
    }

    fn batch_calculate_internal(&self, options: Vec<OptionParams>) -> Vec<OptionResult> {
        let mut results = Vec::with_capacity(options.len());

        for opt in options {
            let price = if opt.is_call {
                self.price_call(opt.spot, opt.strike, opt.time, opt.vol, opt.rate, opt.div)
            } else {
                self.price_put(opt.spot, opt.strike, opt.time, opt.vol, opt.rate, opt.div)
            };
            
            let greeks = self.calculate_greeks(opt.spot, opt.strike, opt.time, opt.vol, opt.rate, opt.div);
            
            results.push(OptionResult {
                price,
                greeks,
            });
        }
        results
    }

    pub fn batch_calculate_compact(&self, params: Float64Array) -> Float64Array {
        let input = params.to_vec();
        let stride = 7; // spot, strike, time, vol, rate, div, is_call
        let num_options = input.len() / stride;
        let mut results = Vec::with_capacity(num_options * 6); // price + 5 greeks

        for i in 0..num_options {
            let offset = i * stride;
            let spot = input[offset];
            let strike = input[offset + 1];
            let time = input[offset + 2];
            let vol = input[offset + 3];
            let rate = input[offset + 4];
            let div = input[offset + 5];
            let is_call = input[offset + 6] > 0.5;

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

    fn calculate_d1_d2(&self, spot: f64, strike: f64, time: f64, vol: f64, rate: f64, div: f64) -> (f64, f64) {
        let d1 = ((spot / strike).ln() + (rate - div + 0.5 * vol.powi(2)) * time) / (vol * time.sqrt());
        let d2 = d1 - vol * time.sqrt();
        (d1, d2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_call_price() {
        let bs = BlackScholesWASM::new();
        // Reference: Spot=100, K=100, T=1, V=0.2, R=0.05, D=0.0
        let price = bs.price_call(100.0, 100.0, 1.0, 0.2, 0.05, 0.0);
        // Standard BS price for these params is ~10.45058
        assert!((price - 10.45058).abs() < 1e-4);
    }

    #[test]
    fn test_put_price() {
        let bs = BlackScholesWASM::new();
        let price = bs.price_put(100.0, 100.0, 1.0, 0.2, 0.05, 0.0);
        // Standard BS put price is ~5.57352
        assert!((price - 5.57352).abs() < 1e-4);
    }

    #[test]
    fn test_greeks() {
        let bs = BlackScholesWASM::new();
        let greeks = bs.calculate_greeks(100.0, 100.0, 1.0, 0.2, 0.05, 0.0);
        
        // Ref values for Delta: ~0.6368
        assert!((greeks.delta - 0.6368).abs() < 1e-3);
        // Ref values for Gamma: ~0.01876
        assert!((greeks.gamma - 0.01876).abs() < 1e-4);
        // Ref values for Vega: ~0.3752 (S*nd1*sqrtT / 100)
        assert!((greeks.vega - 0.3752).abs() < 1e-3);
        // Ref values for Theta: ~-0.01757
        assert!((greeks.theta - -0.01757).abs() < 1e-4);
        // Ref values for Rho: ~0.53232
        assert!((greeks.rho - 0.53232).abs() < 1e-3);
    }

    #[test]
    fn test_edge_cases() {
        let bs = BlackScholesWASM::new();
        
        // T = 0 (Expiry)
        assert_eq!(bs.price_call(100.0, 90.0, 0.0, 0.2, 0.05, 0.0), 10.0);
        assert_eq!(bs.price_call(100.0, 110.0, 0.0, 0.2, 0.05, 0.0), 0.0);
        assert_eq!(bs.price_put(100.0, 110.0, 0.0, 0.2, 0.05, 0.0), 10.0);
        assert_eq!(bs.price_put(100.0, 90.0, 0.0, 0.2, 0.05, 0.0), 0.0);

        // Very high volatility
        let price_high_vol = bs.price_call(100.0, 100.0, 1.0, 5.0, 0.05, 0.0);
        assert!(price_high_vol > 90.0); // Should be close to spot as vol -> infinity

        // Deep in the money
        let price_itm = bs.price_call(100.0, 10.0, 1.0, 0.2, 0.05, 0.0);
        assert!((price_itm - (100.0 - 10.0 * (-0.05 * 1.0f64).exp())).abs() < 1e-2);
    }

    #[test]
    fn test_iv_solver() {
        let bs = BlackScholesWASM::new();
        let spot = 100.0;
        let strike = 100.0;
        let time = 1.0;
        let rate = 0.05;
        let div = 0.0;
        let target_vol = 0.2;
        
        let price = bs.price_call(spot, strike, time, target_vol, rate, div);
        let solved_vol = bs.solve_iv(price, spot, strike, time, rate, div, true);
        
        assert!((solved_vol - target_vol).abs() < 1e-4);
    }

    #[test]
    fn test_batch_calculate() {
        let bs = BlackScholesWASM::new();
        let options = vec![
            OptionParams { spot: 100.0, strike: 100.0, time: 1.0, vol: 0.2, rate: 0.05, div: 0.0, is_call: true },
            OptionParams { spot: 100.0, strike: 100.0, time: 1.0, vol: 0.2, rate: 0.05, div: 0.0, is_call: false },
        ];
        
        let results = bs.batch_calculate_internal(options);
        
        assert_eq!(results.len(), 2);
        assert!((results[0].price - 10.45058).abs() < 1e-4);
        assert!((results[1].price - 5.57352).abs() < 1e-4);
    }
}