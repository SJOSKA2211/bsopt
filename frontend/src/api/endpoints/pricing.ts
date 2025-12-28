import apiClient from '@/utils/apiClient';

/**
 * Types for Pricing API
 */

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
  computation_time_ms?: number;
}

export interface BatchPriceRequest {
  options: PriceRequest[];
}

export interface BatchPriceResponse {
  results: PriceResponse[];
  total_count: number;
  computation_time_ms: number;
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

export interface IVRequest {
  spot: number;
  strike: number;
  time_to_expiry: number;
  rate: number;
  option_price: number;
  option_type: 'call' | 'put';
  dividend_yield?: number;
}

export interface IVResponse {
  implied_volatility: number;
  option_price: number;
  spot: number;
  strike: number;
  iterations: number;
  converged: boolean;
}

/**
 * Pricing API Endpoints
 */

export const priceOption = async (data: PriceRequest): Promise<PriceResponse> => {
  const response = await apiClient.post<PriceResponse>('/pricing/price', data);
  return response.data;
};

export const batchPriceOptions = async (data: BatchPriceRequest): Promise<BatchPriceResponse> => {
  const response = await apiClient.post<BatchPriceResponse>('/pricing/batch', data);
  return response.data;
};

export const calculateGreeks = async (data: GreeksRequest): Promise<GreeksResponse> => {
  const response = await apiClient.post<GreeksResponse>('/pricing/greeks', data);
  return response.data;
};

export const calculateIV = async (data: IVRequest): Promise<IVResponse> => {
  const response = await apiClient.post<IVResponse>('/pricing/implied-volatility', data);
  return response.data;
};
