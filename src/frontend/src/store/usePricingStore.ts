import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

interface PriceData {
  symbol: string;
  price: number;
  delta: number;
  gamma: number;
  timestamp: number;
}

interface PricingState {
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
  subscribeWithSelector((set) => ({
    prices: {},
    systemGamma: 0,
    mlAccuracy: 98.2,
    portfolioTotal: 1248392.42,
    
    updatePrice: (symbol, data) => 
      set((state) => ({
        prices: {
          ...state.prices,
          [symbol]: {
            ...(state.prices[symbol] || { symbol, price: 0, delta: 0, gamma: 0, timestamp: Date.now() }),
            ...data,
            timestamp: Date.now(),
          },
        },
      })),

    batchUpdate: (updates) =>
      set((state) => {
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
      
    setGlobalMetrics: (metrics) =>
      set((state) => ({
         ...state,
         ...metrics,
      })),
  }))
);
