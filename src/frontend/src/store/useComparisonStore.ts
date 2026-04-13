import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

export interface ComparisonMetrics {
  userPnl: number;
  aiPnl: number;
  userSharpe: number;
  aiSharpe: number;
  userWinRate: number;
  aiWinRate: number;
  timestamp: number;
}

interface ComparisonState {
  metrics: ComparisonMetrics;
  mode: 'realtime' | 'historical';
  modelsSelected: string[];
  setMetrics: (metrics: Partial<ComparisonMetrics>) => void;
  setMode: (mode: 'realtime' | 'historical') => void;
  toggleModel: (model: string) => void;
}

// State management for User vs AI Comparison
export const useComparisonStore = create<ComparisonState>()(
  subscribeWithSelector((set) => ({
    metrics: {
      userPnl: 0,
      aiPnl: 0,
      userSharpe: 0,
      aiSharpe: 0,
      userWinRate: 0,
      aiWinRate: 0,
      timestamp: Date.now(),
    },
    mode: 'realtime',
    modelsSelected: ['DeepHedge Model', 'Quant-RL'],
    
    setMetrics: (data) => 
      set((state) => ({
        metrics: {
          ...state.metrics,
          ...data,
          timestamp: Date.now(),
        },
      })),

    setMode: (mode) => set({ mode }),
    
    toggleModel: (model) => set((state) => {
      const exists = state.modelsSelected.includes(model);
      return {
        modelsSelected: exists 
          ? state.modelsSelected.filter(m => m !== model)
          : [...state.modelsSelected, model]
      };
    }),
  }))
);
