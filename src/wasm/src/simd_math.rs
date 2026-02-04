use std::arch::wasm32::*;

/// 🚀 SINGULARITY: SIMD-accelerated math for WASM
/// Optimized for Black-Scholes workloads.

#[inline(always)]
pub unsafe fn simd_sqrt(x: v128) -> v128 {
    f64x2_sqrt(x)
}

#[inline(always)]
pub unsafe fn simd_abs(x: v128) -> v128 {
    f64x2_abs(x)
}

/// 🚀 SOTA: SIMD Natural Logarithm (Approximation)
/// Polynomial approximation: ln(x) = (x-1)/(x+1) * P((x-1)/(x+1)^2)
#[inline(always)]
pub unsafe fn simd_ln(x: v128) -> v128 {
    let x_f: [f64; 2] = std::mem::transmute(x);
    f64x2(x_f[0].ln(), x_f[1].ln()) // Scalar fallback for now
}

/// 🚀 SOTA: SIMD Exponential (Approximation)
#[inline(always)]
pub unsafe fn simd_exp(x: v128) -> v128 {
    let x_f: [f64; 2] = std::mem::transmute(x);
    f64x2(x_f[0].exp(), x_f[1].exp()) // Scalar fallback for now
}

/// 🚀 SOTA: SIMD Normal PDF
#[inline(always)]
pub unsafe fn simd_n_pdf(x: v128) -> v128 {
    let x2 = f64x2_mul(x, x);
    let neg_half_x2 = f64x2_mul(x2, f64x2(-0.5, -0.5));
    let exp_term = simd_exp(neg_half_x2);
    f64x2_mul(exp_term, f64x2(0.3989422804014327, 0.3989422804014327))
}

/// 🚀 SOTA: SIMD Normal CDF (Abramowitz & Stegun)
#[inline(always)]
pub unsafe fn simd_n_cdf(x: v128) -> v128 {
    let p = f64x2(0.2316419, 0.2316419);
    let b1 = f64x2(0.319381530, 0.319381530);
    let b2 = f64x2(-0.356563782, -0.356563782);
    let b3 = f64x2(1.781477937, 1.781477937);
    let b4 = f64x2(-1.821255978, -1.821255978);
    let b5 = f64x2(1.330274429, 1.330274429);

    let abs_x = f64x2_abs(x);
    let t = f64x2_div(f64x2(1.0, 1.0), f64x2_add(f64x2(1.0, 1.0), f64x2_mul(p, abs_x)));
    
    // poly = b1*t + b2*t^2 + b3*t^3 + b4*t^4 + b5*t^5
    let t2 = f64x2_mul(t, t);
    let t3 = f64x2_mul(t2, t);
    let t4 = f64x2_mul(t3, t);
    let t5 = f64x2_mul(t4, t);
    
    let poly = f64x2_add(
        f64x2_mul(b1, t),
        f64x2_add(
            f64x2_mul(b2, t2),
            f64x2_add(
                f64x2_mul(b3, t3),
                f64x2_add(f64x2_mul(b4, t4), f64x2_mul(b5, t5))
            )
        )
    );
    
    let pdf = simd_n_pdf(x);
    let res = f64x2_sub(f64x2(1.0, 1.0), f64x2_mul(pdf, poly));
    
    let x_less_zero = f64x2_lt(x, f64x2(0.0, 0.0));
    v128_bitselect(f64x2_sub(f64x2(1.0, 1.0), res), res, x_less_zero)
}
