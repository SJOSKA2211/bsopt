use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use statrs::distribution::{Normal, Continuous, ContinuousCDF};

#[wasm_bindgen]
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
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
}
