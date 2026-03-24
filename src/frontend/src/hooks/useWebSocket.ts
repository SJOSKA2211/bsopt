// src/frontend/src/hooks/useWebSocket.ts (Optimized)
import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
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
  const [error, setError] = useState<Error | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectCountRef = useRef(0);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const bufferRef = useRef<T | null>(null);
  const lastUpdateRef = useRef<number>(0);
  const isMountedRef = useRef(true);

  const symbolsString = useMemo(() => options.symbols.join(','), [options.symbols]);
  
  const connect = useCallback(() => {
    if (!isMountedRef.current || !options.enabled) return;

    // Clean up existing connection
    if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
    }

    try {
        const ws = new WebSocket(options.url);
        if (options.useProtobuf) ws.binaryType = 'arraybuffer';
        wsRef.current = ws;

        ws.onopen = () => {
            if (!isMountedRef.current) return;
            setIsConnected(true);
            setError(null);
            reconnectCountRef.current = 0;
            ws.send(JSON.stringify({ type: 'subscribe', symbols: options.symbols }));
        };

        ws.onmessage = (event) => {
            if (!isMountedRef.current) return;
            try {
                const parsed = JSON.parse(event.data) as T;
                bufferRef.current = parsed;
                
                const updateInterval = 1000 / (options.updateFrequency || 10);
                const now = performance.now();
                if (now - lastUpdateRef.current > updateInterval) {
                    setData(bufferRef.current);
                    lastUpdateRef.current = now;
                }
            } catch (e) {
                console.error('[WebSocket] Parse error:', e);
            }
        };

        ws.onclose = (event) => {
            if (!isMountedRef.current) return;
            setIsConnected(false);
            wsRef.current = null;

            if (options.enabled) {
                const backoff = Math.min(1000 * Math.pow(2, reconnectCountRef.current), 30000);
                console.log(`[WebSocket] Reconnecting in ${backoff}ms...`);
                reconnectCountRef.current += 1;
                reconnectTimeoutRef.current = setTimeout(connect, backoff);
            }
        };

        ws.onerror = (err) => {
            if (!isMountedRef.current) return;
            setError(new Error('WebSocket connection error'));
            console.error('[WebSocket] Error:', err);
        };
    } catch (err) {
        console.error('[WebSocket] Failed to connect:', err);
        setError(err as Error);
    }
  }, [options.url, options.enabled, options.symbols, options.useProtobuf, options.updateFrequency]);

  useEffect(() => {
    isMountedRef.current = true;
    connect();

    return () => {
        isMountedRef.current = false;
        if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
    };
  }, [connect]);

  const sendMessage = useCallback((msg: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  return { data, isConnected, error, sendMessage };
}
