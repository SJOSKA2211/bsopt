import { renderHook, act, waitFor } from '@testing-library/react';
import { useWebSocket } from '../src/hooks/useWebSocket';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import { Server, WebSocket as MockWebSocket } from 'mock-socket';

// Ensure WebSocket is mocked globally
if (typeof global !== 'undefined') {
  (global as any).WebSocket = MockWebSocket;
}
if (typeof window !== 'undefined') {
  (window as any).WebSocket = MockWebSocket;
}

describe('useWebSocket', () => {
  const WS_URL = 'ws://localhost:8080';
  let mockServer: Server;

  it('verifies websocket mock', () => {
    expect(window.WebSocket).toBe(MockWebSocket);
    expect(new WebSocket(WS_URL)).toBeInstanceOf(MockWebSocket);
  });

  beforeEach(() => {
    vi.useFakeTimers();
    mockServer = new Server(WS_URL);
  });

  afterEach(() => {
    mockServer.stop();
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it('connects and receives data', async () => {
    const { result } = renderHook(() => useWebSocket({ url: WS_URL }));

    // Allow connection to establish under fake timers
    await act(async () => {
      vi.runAllTimers();
    });

    // Wait for connection
    await waitFor(() => expect(result.current.isConnected).toBe(true));

    const testData = { type: 'TICK', price: 100 };
    
    await act(async () => {
      mockServer.emit('message', JSON.stringify(testData));
    });

    await waitFor(() => {
      expect(result.current.lastMessage).toEqual(testData);
    });
  });

  it('handles reconnection', async () => {
    const { result } = renderHook(() => useWebSocket({ url: WS_URL }));

    // 1. Initial connection
    await act(async () => {
      vi.advanceTimersByTime(100);
    });
    await waitFor(() => expect(result.current.isConnected).toBe(true));

    // 2. Simulate disconnect
    await act(async () => {
      mockServer.close(); // Closes all connections
    });

    await waitFor(() => expect(result.current.isConnected).toBe(false));

    // 3. Wait for backoff and reconnection
    // The hook has a 1000ms base backoff
    await act(async () => {
      vi.advanceTimersByTime(1100);
    });

    // 4. Wait for connection to be re-established
    await waitFor(() => expect(result.current.isConnected).toBe(true), { timeout: 5000 });
  });

  it('cleans up on unmount', async () => {
    const { result, unmount } = renderHook(() => useWebSocket({ url: WS_URL }));
    
    await waitFor(() => expect(result.current.isConnected).toBe(true));
    
    unmount();
  });
});
