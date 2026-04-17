import { useState, useEffect, useRef } from 'react';

interface WebSocketOptions {
  url: string;
  reconnectInterval?: number;
}

export const useWebSocket = ({ url, reconnectInterval = 1000 }: WebSocketOptions) => {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<any>(null);

  useEffect(() => {
    let isMounted = true;

    const connect = () => {
      // Clear any existing timeout
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }

      const socket = new WebSocket(url);
      socketRef.current = socket;

      socket.onopen = () => {
        if (isMounted) setIsConnected(true);
      };

      socket.onmessage = (event) => {
        if (isMounted) {
          try {
            const data = JSON.parse(event.data);
            setLastMessage(data);
          } catch (e) {
            setLastMessage(event.data);
          }
        }
      };

      socket.onclose = () => {
        if (isMounted) {
          setIsConnected(false);
          // Auto-reconnect following backoff
          reconnectTimeoutRef.current = setTimeout(connect, reconnectInterval);
        }
      };

      socket.onerror = () => {
        if (isMounted) {
          setIsConnected(false);
        }
      };
    };

    connect();

    return () => {
      isMounted = false;
      if (socketRef.current) {
        socketRef.current.onclose = null; // Prevent reconnect on unmount
        socketRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [url, reconnectInterval]);

  return { isConnected, lastMessage };
};
