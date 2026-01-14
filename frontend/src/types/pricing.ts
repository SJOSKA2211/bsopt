export interface PriceRequest {
  spot: number;
  strike: number;
  time_to_expiry: number;
  rate: number;
  volatility: number;
  option_type: 'call' | 'put';
  dividend_yield?: number;
  model?: 'black_scholes' | 'monte_carlo' | 'binomial';
}

export interface PriceResponse {
  price: number;
  spot: number;
  strike: number;
  time_to_expiry: number;
  rate: number;
  volatility: number;
  option_type: string;
  model: string;
  timestamp: string;
  cached?: boolean;
  computation_time_ms?: number;
}

export interface BatchPriceRequest {
  options: PriceRequest[];
}

export interface BatchPriceResponse {
  results: PriceResponse[];
  total_count: number;
  computation_time_ms: number;
  cached_count?: number;
}

export interface GreeksRequest {
  spot: number;
  strike: number;
  time_to_expiry: number;
  rate: number;
  volatility: number;
  option_type: 'call' | 'put';
  dividend_yield?: number;
}

export interface GreeksResponse {
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  rho: number;
  option_price: number;
  spot: number;
  strike: number;
  time_to_expiry: number;
  volatility: number;
  option_type: string;
  timestamp: string;
}

export interface ImpliedVolatilityRequest {
  spot: number;
  strike: number;
  time_to_expiry: number;
  rate: number;
  option_price: number;
  option_type: 'call' | 'put';
  dividend_yield?: number;
}

export interface ImpliedVolatilityResponse {
  implied_volatility: number;
  option_price: number;
  spot: number;
  strike: number;
  iterations: number;
  converged: boolean;
}

export interface ExoticPriceRequest {
  spot: number;
  strike: number;
  time_to_expiry: number;
  rate: number;
  volatility: number;
  option_type?: 'call' | 'put';
  dividend_yield?: number;
  exotic_type: 'asian' | 'barrier' | 'lookback' | 'digital';
  
  // Specific params
  barrier?: number;
  rebate?: number;
  barrier_type?: string;
  asian_type?: string;
  strike_type?: string;
  n_observations?: number;
  payout?: number;
}

export interface ExoticPriceResponse {
  price: number;
  confidence_interval?: number[];
  exotic_type: string;
  timestamp: string;
}
