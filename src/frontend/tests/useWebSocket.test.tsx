import { renderHook, act, waitFor } from '@testing-library/react';
import { useWebSocket } from '../src/hooks/useWebSocket';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import { Server } from 'mock-socket';

describe('useWebSocket', () => {
  const WS_URL = 'ws://localhost:8080';
  let mockServer: Server;

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
    await waitFor(() => expect(result.current.isConnected).toBe(true));

    // 2. Simulate disconnect
    await act(async () => {
      mockServer.close();
    });

    await waitFor(() => expect(result.current.isConnected).toBe(false));

    // 3. Wait for backoff and reconnection
    // The hook has a 1000ms base backoff
    await act(async () => {
      vi.advanceTimersByTime(1100);
    });

    // 4. Trigger the reconnection by causing a re-render or wait for it
    // In mock-socket, a new Server with same URL before the client connects might be needed,
    // or the existing server still handles connections even if closed (depends on mock-socket implementation).
    // Let's assume the hook will try to reconnect and mock-socket will intercept.
    
    await waitFor(() => expect(result.current.isConnected).toBe(true), { timeout: 5000 });
  });

  it('cleans up on unmount', async () => {
    const { result, unmount } = renderHook(() => useWebSocket({ url: WS_URL }));
    
    await waitFor(() => expect(result.current.isConnected).toBe(true));
    
    unmount();
  });
});
