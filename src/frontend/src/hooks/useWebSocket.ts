// src/frontend/src/hooks/useWebSocket.ts (Optimized)
import { useState, useEffect, useRef, useMemo } from 'react';
// import { protobuf } from 'protobufjs'; // Removed unused import causing build error

/**
 * Derives a WebSocket URL from the current page origin.
 * Uses wss:// for HTTPS pages, ws:// otherwise.
 * Falls back to a provided URL if not in a browser context.
 */
export function getWebSocketUrl(path: string): string {
  if (typeof window === 'undefined') return `ws://localhost:8000${path}`;
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}${path}`;
}

interface WebSocketHookOptions {
  url: string;
  symbols: string[];
  enabled: boolean;
  useProtobuf?: boolean;
  updateFrequency?: number; // Hz
}

export function useWebSocket<T>(options: WebSocketHookOptions) {
  const [data, setData] = useState<T | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const protoRootRef = useRef<unknown>(null);
  const bufferRef = useRef<T | null>(null);
  const lastUpdateRef = useRef<number>(0);

  const symbolsString = useMemo(() => options.symbols.join(','), [options.symbols]);
  
  useEffect(() => {
    if (!options.enabled) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setIsConnected(false);
      return;
    }

    const updateInterval = 1000 / (options.updateFrequency || 10);
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let isMounted = true;

    const connect = () => {
      if (!isMounted || !options.enabled) return;

      const ws = new WebSocket(options.url);
      if (options.useProtobuf) ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      ws.onopen = () => {
        if (!isMounted) return;
        setIsConnected(true);
        ws.send(JSON.stringify({ type: 'subscribe', symbols: options.symbols }));
      };

      ws.onmessage = (event) => {
        if (!isMounted) return;
        try {
          let parsed: T;
          if (options.useProtobuf && protoRootRef.current) {
            const MessageType = (protoRootRef.current as any).lookupType('bsopt.OptionsData');
            const decoded = MessageType.decode(new Uint8Array(event.data));
            parsed = MessageType.toObject(decoded) as T;
          } else {
            parsed = JSON.parse(event.data);
          }

          bufferRef.current = parsed;
          const now = performance.now();
          if (now - lastUpdateRef.current > updateInterval) {
            setData(bufferRef.current);
            lastUpdateRef.current = now;
          }
        } catch (e) {
          console.error('WS_PARSE_ERROR', e);
        }
      };

      ws.onclose = () => {
        if (!isMounted) return;
        setIsConnected(false);
        wsRef.current = null;
        if (options.enabled) {
          reconnectTimer = setTimeout(connect, 1000); // 1s reconnect
        }
      };

      ws.onerror = () => {
        if (!isMounted) return;
        ws.close();
      };
    };

    connect();

    return () => {
      isMounted = false;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [options.url, options.enabled, symbolsString, options.symbols, options.useProtobuf, options.updateFrequency]);

  return { data, isConnected };
}
