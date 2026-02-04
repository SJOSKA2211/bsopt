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
  updatePrice: (symbol: string, data: Partial<PriceData>) => void;
  batchUpdate: (updates: Record<string, Partial<PriceData>>) => void;
}

// 🚀 SINGULARITY: Zero-re-render transient state store
export const usePricingStore = create<PricingState>()(
  subscribeWithSelector((set) => ({
    prices: {},
    
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
  }))
);
