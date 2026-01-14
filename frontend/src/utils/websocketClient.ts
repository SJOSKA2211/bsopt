import { useState, useEffect, useRef, useCallback } from 'react';
import { WebSocketMessagePayload } from '../types/websocket';

interface WebSocketClientOptions {
  onMessage?: (message: WebSocketMessagePayload) => void;
  onOpen?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (event: Event) => void;
  autoConnect?: boolean;
  reconnectInterval?: number; // milliseconds
  maxReconnectAttempts?: number;
}

const useWebSocket = (url: string, options: WebSocketClientOptions = {}) => {
  const {
    onMessage,
    onOpen,
    onClose,
    onError,
    autoConnect = true,
    reconnectInterval = 5000, // 5 seconds
    maxReconnectAttempts = 10,
  } = options;

  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [latestMessage, setLatestMessage] = useState<WebSocketMessagePayload | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef<number>(0);
  const reconnectTimerRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    // Define attemptReconnect inside connect to avoid circular dependency issues with useCallback
    const attemptReconnect = () => {
      if (reconnectTimerRef.current) return; // Already scheduled

      if (reconnectAttemptsRef.current < maxReconnectAttempts) {
        reconnectAttemptsRef.current += 1;
        console.log(`Attempting to reconnect to WebSocket (${reconnectAttemptsRef.current}/${maxReconnectAttempts})...`);
        reconnectTimerRef.current = setTimeout(() => {
          connect(); // Calls the outer connect, correctly closing over it
          reconnectTimerRef.current = null; // Clear timer ref after execution
        }, reconnectInterval);
      } else {
        console.error('Max WebSocket reconnect attempts reached.');
      }
    };

    wsRef.current = new WebSocket(url);

    wsRef.current.onopen = (event) => {
      console.log('WebSocket connected:', url);
      setIsConnected(true);
      reconnectAttemptsRef.current = 0; // Reset attempts on successful connection
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      if (onOpen) {
        onOpen(event);
      }
    };

    wsRef.current.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        setLatestMessage(message);
        if (onMessage) {
          onMessage(message);
        }
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
        setLatestMessage(event.data); // Keep raw data if parsing fails
        if (onMessage) {
          onMessage(event.data); // Pass raw data if parsing fails
        }
      }
    };

    wsRef.current.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);
      setIsConnected(false);
      if (onClose) {
        onClose(event);
      }
      // Attempt to reconnect if autoConnect is true and not intentionally closed
      if (autoConnect && !event.wasClean) {
        attemptReconnect();
      }
    };

    wsRef.current.onerror = (event) => {
      console.error('WebSocket error:', event);
      if (onError) {
        onError(event);
      }
      // Error often leads to close, so reconnect attempt is usually handled by onclose
    };
  }, [url, onOpen, onMessage, onClose, onError, autoConnect, reconnectInterval, maxReconnectAttempts]); // attemptReconnect no longer a dependency

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      setIsConnected(false);
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      reconnectAttemptsRef.current = 0; // Reset attempts
    }
  }, []);

  const sendMessage = useCallback((message: WebSocketMessagePayload) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify(message));
      } catch (e) {
        console.error('Failed to send WebSocket message:', e);
      }
    } else {
      console.warn('WebSocket is not connected. Cannot send message.');
    }
  }, []);

  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      // Cleanup on unmount
      disconnect();
    };
  }, [connect, disconnect, autoConnect]);

  return {
    isConnected,
    latestMessage,
    connect,
    disconnect,
    sendMessage,
  };
};

export default useWebSocket;