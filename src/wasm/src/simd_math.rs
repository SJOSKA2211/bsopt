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

/// 🚀 SOTA: SIMD Natural Logarithm (Polynomial Approximation)
#[inline(always)]
pub unsafe fn simd_ln(x: v128) -> v128 {
    let one = f64x2(1.0, 1.0);
    let num = f64x2_sub(x, one);
    let den = f64x2_add(x, one);
    let t = f64x2_div(num, den);
    let t2 = f64x2_mul(t, t);
    
    // t * (2.0 + (2.0/3.0)*t^2 + (2.0/5.0)*t^4)
    f64x2_mul(t, f64x2_add(f64x2(2.0, 2.0), f64x2_mul(t2, f64x2_add(f64x2(0.66666666, 0.66666666), f64x2_mul(t2, f64x2(0.4, 0.4))))))
}

/// 🚀 SOTA: SIMD Exponential (Polynomial Approximation)
#[inline(always)]
pub unsafe fn simd_exp(x: v128) -> v128 {
    let one = f64x2(1.0, 1.0);
    let x2 = f64x2_mul(x, x);
    let x3 = f64x2_mul(x2, x);
    let x4 = f64x2_mul(x2, x2);
    
    f64x2_add(one, f64x2_add(x, f64x2_add(f64x2_mul(x2, f64x2(0.5, 0.5)), f64x2_add(f64x2_mul(x3, f64x2(0.16666666, 0.16666666)), f64x2_mul(x4, f64x2(0.04166666, 0.04166666))))))
}

/// 🚀 SOTA: SIMD Normal PDF
#[inline(always)]
pub unsafe fn simd_n_pdf(x: v128) -> v128 {
    let inv_sqrt_2pi = f64x2(0.3989422804014327, 0.3989422804014327);
    let neg_half = f64x2(-0.5, -0.5);
    let arg = f64x2_mul(neg_half, f64x2_mul(x, x));
    f64x2_mul(inv_sqrt_2pi, simd_exp(arg))
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