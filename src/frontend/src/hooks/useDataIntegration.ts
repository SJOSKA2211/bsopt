import { useEffect } from 'react';
import { usePricingStore } from '../store/usePricingStore';
import { useWebSocket, getWebSocketUrl } from './useWebSocket';

interface MarketDataPoint {
  symbol: string;
  price: number;
  bid: number;
  ask: number;
  volume: number;
  timestamp: number;
}

interface MarketUpdate {
  type: string;
  payload: MarketDataPoint | MarketDataPoint[];
}

interface UseDataIntegrationOptions {
  symbols: string[];
  enabled?: boolean;
}

export function useDataIntegration({ symbols, enabled = true }: UseDataIntegrationOptions) {
  const batchUpdate = usePricingStore((state) => state.batchUpdate);
  const updatePrice = usePricingStore((state) => state.updatePrice);

  const { data, isConnected } = useWebSocket<MarketUpdate>({
    url: getWebSocketUrl('/api/v1/ws/market'),
    symbols,
    enabled,
    updateFrequency: 20, // 20Hz for high-perf updates
  });

  useEffect(() => {
    if (data && data.type === 'market_update' && data.payload) {
      if (Array.isArray(data.payload)) {
        const updates: Record<string, { price: number; timestamp: number }> = {};
        data.payload.forEach((tick) => {
          updates[tick.symbol] = { price: tick.price, timestamp: tick.timestamp };
        });
        batchUpdate(updates);
      } else {
        updatePrice(data.payload.symbol, { 
          price: data.payload.price, 
          timestamp: data.payload.timestamp 
        });
      }
    }
  }, [data, batchUpdate, updatePrice]);

  return { isConnected };
}
