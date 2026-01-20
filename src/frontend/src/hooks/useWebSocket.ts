// src/frontend/src/hooks/useWebSocket.ts
import { useState, useEffect, useRef } from 'react';

interface WebSocketHookOptions {
  url: string;
  symbols: string[];
  enabled: boolean;
}

export function useWebSocket<T>(options: WebSocketHookOptions) {
  const [data, setData] = useState<T | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (!options.enabled) {
      return;
    }

    let isComponentMounted = true;

    const connect = () => {
      if (!isComponentMounted) return;

      // Close existing connection if any
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
      }

      const ws = new WebSocket(options.url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!isComponentMounted) return;
        setIsConnected(true);
        console.log('WebSocket connected');
        // Send subscription messages for symbols
        ws.send(JSON.stringify({ type: 'subscribe', symbols: options.symbols }));
      };

      ws.onmessage = (event) => {
        if (!isComponentMounted) return;
        try {
          const parsedData: T = JSON.parse(event.data);
          setData(parsedData);
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        if (!isComponentMounted) return;
        console.log('WebSocket disconnected. Attempting to reconnect...');
        reconnectTimeoutRef.current = setTimeout(connect, 3000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        // onerror is usually followed by onclose
      };
    };

    connect();

    return () => {
      isComponentMounted = false;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [options.url, options.enabled, options.symbols.join(',')]);

  return { data, isConnected };
}
