/**
 * Institutional-Grade Type Definitions for BS-OPT
 * Unified between Frontend (React/TS) and Backend (Python/Pydantic/Strawberry)
 */

export interface MarketData {
  symbol: string;
  last_price: number;
  bid?: number;
  ask?: number;
  volume?: number;
  timestamp: string; // ISO String
}

export interface MLPrediction {
  id: string;
  symbol: string;
  predicted_price: number;
  actual_price?: number;
  prediction_error?: number;
  confidence_interval?: number;
  drift?: number;
  model_name: string;
  timestamp: string;
  last_updated: string;
}

export interface Option {
  id: string;
  symbol: string;
  expiry: string;
  strike: number;
  type: 'CALL' | 'PUT';
  bid: number;
  ask: number;
  last_price: number;
  volume: number;
  open_interest: number;
  implied_volatility: number;
  greeks: Greeks;
}

export interface Greeks {
  delta: number;
  gamma: number;
  vega: number;
  theta: number;
  rho: number;
}

export interface OptionConnection {
  edges: {
    cursor: string;
    node: Option;
  }[];
  pageInfo: {
    hasNextPage: boolean;
    endCursor: string | null;
  };
}

// User & Portfolio Types
export interface User {
  id: string;
  email: string;
  role: 'admin' | 'trader' | 'viewer';
  created_at: string;
}

export interface OptionChainRow {
  id: string;
  strike: number;
  expiry: string;
  underlying_price: number;
  call_bid: number;
  call_ask: number;
  call_last: number;
  call_volume: number;
  call_oi: number;
  call_iv: number;
  call_delta: number;
  call_gamma: number;
  call_theor?: number;
  put_bid: number;
  put_ask: number;
  put_last: number;
  put_volume: number;
  put_oi: number;
  put_iv: number;
  put_delta: number;
  put_gamma: number;
  put_theor?: number;
}
