import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

interface PriceData {
  symbol: string;
  price: number;
  delta: number;
  gamma: number;
  timestamp: number;
  // Extended market data fields (optional, from live data provider)
  prev_close?: number;
  iv_rank?: number;
  hv30?: number;
  put_call_ratio?: number;
  volume?: number;
  open_interest?: number;
  high?: number;
  low?: number;
}

export interface PricingState {
  prices: Record<string, PriceData>;
  systemGamma: number;
  mlAccuracy: number;
  portfolioTotal: number;
  updatePrice: (symbol: string, data: Partial<PriceData>) => void;
  batchUpdate: (updates: Record<string, Partial<PriceData>>) => void;
  setGlobalMetrics: (metrics: { systemGamma?: number; mlAccuracy?: number; portfolioTotal?: number }) => void;
}

// Zero-re-render transient state store
export const usePricingStore = create<PricingState>()(
  subscribeWithSelector((set: any) => ({
    prices: {},
    systemGamma: 0,
    mlAccuracy: 0,
    portfolioTotal: 0,
    
    updatePrice: (symbol: string, data: Partial<PriceData>) => 
      set((state: PricingState) => ({
        prices: {
          ...state.prices,
          [symbol]: {
            ...(state.prices[symbol] || { symbol, price: 0, delta: 0, gamma: 0, timestamp: Date.now() }),
            ...data,
            timestamp: Date.now(),
          },
        },
      })),

    batchUpdate: (updates: Record<string, Partial<PriceData>>) =>
      set((state: PricingState) => {
        const newPrices = { ...state.prices };
        for (const [symbol, data] of Object.entries(updates)) {
          newPrices[symbol] = {
            ...(newPrices[symbol] || { symbol, price: 0, delta: 0, gamma: 0, timestamp: Date.now() }),
            ...data,
            timestamp: Date.now(),
          };
        }
        return { prices: newPrices };
      }),
      
    setGlobalMetrics: (metrics: { systemGamma?: number; mlAccuracy?: number; portfolioTotal?: number }) =>
      set((state: PricingState) => ({
         ...state,
         ...metrics,
      })),
  }))
);
