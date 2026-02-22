// src/frontend/src/hooks/useWebSocket.ts (Optimized)
import { useState, useEffect, useRef } from 'react';
// @ts-ignore
import * as protobuf from 'protobufjs';

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
  const protoRootRef = useRef<any>(null);
  const bufferRef = useRef<T | null>(null);
  const lastUpdateRef = useRef<number>(0);

  useEffect(() => {
    if (!options.enabled) return;

    const updateInterval = 1000 / (options.updateFrequency || 10); // Default 10Hz

    const connect = () => {
      const ws = new WebSocket(options.url);
      if (options.useProtobuf) ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        ws.send(JSON.stringify({ type: 'subscribe', symbols: options.symbols }));
      };

      ws.onmessage = (event) => {
        try {
          let parsed: T;
          if (options.useProtobuf && protoRootRef.current) {
            const MessageType = protoRootRef.current.lookupType('bsopt.OptionsData');
            const decoded = MessageType.decode(new Uint8Array(event.data));
            parsed = MessageType.toObject(decoded) as T;
          } else {
            parsed = JSON.parse(event.data);
          }

          // OPTIMIZED: Throttled State Dispatch
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
        setIsConnected(false);
        setTimeout(connect, 3000);
      };
    };

    connect();
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, [options.url, options.enabled, options.symbols.join(','), options.useProtobuf, options.updateFrequency]);

  return { data, isConnected };
}
