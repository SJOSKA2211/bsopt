/* tslint:disable */
/* eslint-disable */

export class AmericanOptionsWASM {
  free(): void;
  [Symbol.dispose](): void;
  constructor();
  /**
   * Longstaff-Schwartz (LSM) implementation for American Options in WASM.
   */
  price_lsm(spot: number, strike: number, time: number, vol: number, rate: number, div: number, is_call: boolean, num_paths: number, num_steps: number): number;
}

export class BlackScholesWASM {
  free(): void;
  [Symbol.dispose](): void;
  price_call(spot: number, strike: number, time: number, vol: number, rate: number, div: number): number;
  price_american(spot: number, strike: number, time: number, vol: number, rate: number, div: number, is_call: boolean, m: number, n: number): number;
  price_heston_mc(spot: number, strike: number, time: number, r: number, v0: number, kappa: number, theta: number, sigma: number, rho: number, is_call: boolean, num_paths: number): number;
  calculate_greeks(spot: number, strike: number, time: number, vol: number, rate: number, div: number): Greeks;
  price_monte_carlo(spot: number, strike: number, time: number, vol: number, rate: number, div: number, is_call: boolean, num_paths: number): number;
  batch_price_heston(params: Float64Array): Float64Array;
  price_american_lsm(spot: number, strike: number, time: number, vol: number, rate: number, div: number, is_call: boolean, num_paths: number, num_steps: number): number;
  batch_calculate_soa(params: any): any;
  /**
   * SIMD-accelerated batch calculation for Black-Scholes.
   * Processes 2 options at a time using f64x2 SIMD (v128) intrinsics.
   */
  batch_calculate_simd(params: Float64Array): Float64Array;
  batch_calculate_view(params: Float64Array): Float64Array;
  batch_price_american(params: Float64Array, m: number, n: number): Float64Array;
  batch_price_monte_carlo(params: Float64Array, num_paths: number): Float64Array;
  /**
   * Highly optimized batch calculation using SIMD, Rayon, and manual prefetching.
   */
  batch_calculate_soa_compact(spots: Float64Array, strikes: Float64Array, times: Float64Array, vols: Float64Array, rates: Float64Array, divs: Float64Array, are_calls: Float64Array): Float64Array;
  constructor();
  solve_iv(price: number, spot: number, strike: number, time: number, rate: number, div: number, is_call: boolean): number;
  price_put(spot: number, strike: number, time: number, vol: number, rate: number, div: number): number;
}

export class CrankNicolsonWASM {
  free(): void;
  [Symbol.dispose](): void;
  price_american(spot: number, strike: number, time: number, vol: number, rate: number, div: number, is_call: boolean, m: number, n: number): number;
  constructor();
}

export class Greeks {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  delta: number;
  gamma: number;
  vega: number;
  theta: number;
  rho: number;
}

export class HestonWASM {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Price European call using Carr-Madan with Simpson's Rule in WASM
   */
  price_call(spot: number, strike: number, time: number, r: number, v0: number, kappa: number, theta: number, sigma: number, rho: number): number;
  price_monte_carlo(spot: number, strike: number, time: number, r: number, v0: number, kappa: number, theta: number, sigma: number, rho: number, is_call: boolean, num_paths: number): number;
  constructor();
}

export class MonteCarloWASM {
  free(): void;
  [Symbol.dispose](): void;
  price_european(spot: number, strike: number, time: number, vol: number, rate: number, div: number, is_call: boolean, num_paths: number, antithetic: boolean): number;
  constructor();
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_americanoptionswasm_free: (a: number, b: number) => void;
  readonly __wbg_blackscholeswasm_free: (a: number, b: number) => void;
  readonly __wbg_get_greeks_delta: (a: number) => number;
  readonly __wbg_get_greeks_gamma: (a: number) => number;
  readonly __wbg_get_greeks_rho: (a: number) => number;
  readonly __wbg_get_greeks_theta: (a: number) => number;
  readonly __wbg_get_greeks_vega: (a: number) => number;
  readonly __wbg_greeks_free: (a: number, b: number) => void;
  readonly __wbg_set_greeks_delta: (a: number, b: number) => void;
  readonly __wbg_set_greeks_gamma: (a: number, b: number) => void;
  readonly __wbg_set_greeks_rho: (a: number, b: number) => void;
  readonly __wbg_set_greeks_theta: (a: number, b: number) => void;
  readonly __wbg_set_greeks_vega: (a: number, b: number) => void;
  readonly americanoptionswasm_price_lsm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => number;
  readonly blackscholeswasm_batch_calculate_simd: (a: number, b: number, c: number) => number;
  readonly blackscholeswasm_batch_calculate_soa: (a: number, b: number, c: number) => void;
  readonly blackscholeswasm_batch_calculate_soa_compact: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number) => number;
  readonly blackscholeswasm_batch_calculate_view: (a: number, b: number, c: number) => number;
  readonly blackscholeswasm_batch_price_american: (a: number, b: number, c: number, d: number, e: number) => number;
  readonly blackscholeswasm_batch_price_heston: (a: number, b: number, c: number) => number;
  readonly blackscholeswasm_batch_price_monte_carlo: (a: number, b: number, c: number, d: number) => number;
  readonly blackscholeswasm_calculate_greeks: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => number;
  readonly blackscholeswasm_new: () => number;
  readonly blackscholeswasm_price_american: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => number;
  readonly blackscholeswasm_price_american_lsm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => number;
  readonly blackscholeswasm_price_call: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => number;
  readonly blackscholeswasm_price_heston_mc: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number) => number;
  readonly blackscholeswasm_price_monte_carlo: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => number;
  readonly blackscholeswasm_price_put: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => number;
  readonly blackscholeswasm_solve_iv: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => number;
  readonly cranknicolsonwasm_price_american: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => number;
  readonly hestonwasm_price_call: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => number;
  readonly hestonwasm_price_monte_carlo: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number) => number;
  readonly montecarlowasm_price_european: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => number;
  readonly __wbg_cranknicolsonwasm_free: (a: number, b: number) => void;
  readonly __wbg_hestonwasm_free: (a: number, b: number) => void;
  readonly __wbg_montecarlowasm_free: (a: number, b: number) => void;
  readonly americanoptionswasm_new: () => number;
  readonly cranknicolsonwasm_new: () => number;
  readonly hestonwasm_new: () => number;
  readonly montecarlowasm_new: () => number;
  readonly __wbindgen_export: (a: number, b: number) => number;
  readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_export3: (a: number) => void;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
