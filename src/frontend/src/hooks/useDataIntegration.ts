import { useEffect, useCallback, useRef } from 'react';

import { usePricingStore } from '../store/usePricingStore';
import { getWebSocketUrl } from './useWebSocket';

interface MarketDataPoint {
  symbol: string;
  price: number;
  bid: number;
  ask: number;
  volume: number;
  timestamp: number;
}

interface UseDataIntegrationOptions {
  symbols: string[];
  enabled?: boolean;
}

export function useDataIntegration({ symbols, enabled = true }: UseDataIntegrationOptions) {
  const batchUpdate = usePricingStore((state) => state.batchUpdate);
  const updatePrice = usePricingStore((state) => state.updatePrice);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const isMountedRef = useRef<boolean>(true);

  // 1. WebSocket for real-time live ticks
  const connectWs = useCallback(() => {
    if (!enabled || symbols.length === 0) return;
    
    // Connect to websocket endpoint
    const wsUrl = getWebSocketUrl('/api/v1/ws/market');
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[DataIntegration] WS Connected, subscribing to:', symbols);
      ws.send(JSON.stringify({ type: 'subscribe', symbols }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'market_update' && data.payload) {
            // Bulk update the transient store for zero-re-render high perf mapping
            if (Array.isArray(data.payload)) {
                const updates: Record<string, any> = {};
                data.payload.forEach((tick: MarketDataPoint) => {
                   updates[tick.symbol] = { price: tick.price, timestamp: tick.timestamp };
                });
                batchUpdate(updates);
            } else {
                updatePrice(data.payload.symbol, { price: data.payload.price, timestamp: data.payload.timestamp });
            }
        }
      } catch (e) {
        console.error('[DataIntegration] WS parse error:', e);
      }
    };

    ws.onclose = () => {
      console.log('[DataIntegration] WS Disconnected');
      if (isMountedRef.current && enabled) {
        // basic backoff/reconnect
        reconnectTimeoutRef.current = setTimeout(connectWs, 2000);
      }
    };

    ws.onerror = (err) => {
        console.error('[DataIntegration] WS Error', err);
        ws.close();
    };

  }, [symbols, enabled, batchUpdate, updatePrice]);

  useEffect(() => {
    isMountedRef.current = true;
    connectWs();

    return () => {
      isMountedRef.current = false;
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectWs]);


  return {
    isConnected: wsRef.current?.readyState === WebSocket.OPEN,
  };
}
