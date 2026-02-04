// src/frontend/src/hooks/useWebSocket.ts
import { useState, useEffect, useRef } from 'react';
import { protobuf } from 'protobufjs'; // Should be installed in frontend dependencies

interface WebSocketHookOptions {
  url: string;
  symbols: string[];
  enabled: boolean;
  useProtobuf?: boolean;
}

export function useWebSocket<T>(options: WebSocketHookOptions) {
  const [data, setData] = useState<T | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const protoRootRef = useRef<any>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!options.enabled) return;

    // Load protobuf schema if enabled
    if (options.useProtobuf) {
      protobuf.load('/schemas.proto', (err, root) => {
        if (!err) protoRootRef.current = root;
      });
    }

    let isComponentMounted = true;

    const connect = () => {
      if (!isComponentMounted) return;

      const ws = new WebSocket(options.url);
      if (options.useProtobuf) {
        ws.binaryType = 'arraybuffer';
      }
      wsRef.current = ws;

      ws.onopen = () => {
        if (!isComponentMounted) return;
        setIsConnected(true);
        ws.send(JSON.stringify({ type: 'subscribe', symbols: options.symbols }));
      };

      ws.onmessage = (event) => {
        if (!isComponentMounted) return;
        try {
          if (options.useProtobuf && protoRootRef.current) {
            const MessageType = protoRootRef.current.lookupType('bsopt.OptionsData');
            const decoded = MessageType.decode(new Uint8Array(event.data));
            setData(MessageType.toObject(decoded) as T);
          } else {
            const parsedData: T = JSON.parse(event.data);
            setData(parsedData);
          }
        } catch (e) {
          console.error('WebSocket parse error:', e);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        if (isComponentMounted) reconnectTimeoutRef.current = setTimeout(connect, 3000);
      };
    };

    connect();
    return () => {
      isComponentMounted = false;
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
      }
    };
  }, [options.url, options.enabled, options.symbols.join(','), options.useProtobuf]);

  return { data, isConnected };
}
