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

  useEffect(() => {
    if (!options.enabled) {
      return;
    }

    const connect = () => {
      const ws = new WebSocket(options.url);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        console.log('WebSocket connected');
        // Send subscription messages for symbols
        ws.send(JSON.stringify({ type: 'subscribe', symbols: options.symbols }));
      };

      ws.onmessage = (event) => {
        const parsedData: T = JSON.parse(event.data);
        setData(parsedData);
      };

      ws.onclose = () => {
        setIsConnected(false);
        console.log('WebSocket disconnected. Attempting to reconnect...');
        // Simple reconnect logic
        setTimeout(connect, 3000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        ws.close();
      };
    };

    connect();

    return () => {
      wsRef.current?.close();
    };
  }, [options.url, options.symbols, options.enabled]);

  return { data, isConnected };
}
