import { renderHook, waitFor } from '@testing-library/react';
import { Server } from 'mock-socket';
import { useWebSocket } from '../src/hooks/useWebSocket';
import { vi, expect, test, describe, afterEach } from 'vitest';

const WS_URL = 'ws://localhost:1234';

describe('useWebSocket', () => {
  let mockServer: Server | null = null;

  afterEach(() => {
    if (mockServer) {
      mockServer.stop();
      mockServer = null;
    }
  });

  test('useWebSocket connects and receives data', async () => {
    mockServer = new Server(WS_URL);
    const mockData = { price: 100, symbol: 'TEST' };
    
    mockServer.on('connection', socket => {
      socket.send(JSON.stringify(mockData));
    });

    const { result } = renderHook(() => useWebSocket({ url: WS_URL, symbols: ['TEST'], enabled: true }));

    // Initially, should not be connected, data should be null
    expect(result.current.isConnected).toBe(false);
    expect(result.current.data).toBe(null);

    // Wait for connection and data reception
    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
      expect(result.current.data).toEqual(mockData);
    }, { timeout: 2000 });
  });

  test('useWebSocket handles reconnection', async () => {
    vi.useFakeTimers();
    mockServer = new Server(WS_URL);
    const mockData = { price: 101, symbol: 'TEST' };
    
    let connectionCount = 0;
    mockServer.on('connection', socket => {
      connectionCount++;
      if (connectionCount === 1) {
        // Force close from server side
        socket.close();
      } else {
        socket.send(JSON.stringify(mockData));
      }
    });

    const { result } = renderHook(() => useWebSocket({ url: WS_URL, symbols: ['TEST'], enabled: true }));

    // Advance for initial connection
    await vi.advanceTimersByTimeAsync(100);

    // Initial connection should happen, then be closed by server immediately
    await waitFor(() => {
      expect(result.current.isConnected).toBe(false);
    });

    // Advance timers to trigger reconnection (default backoff starts at 1000ms)
    await vi.advanceTimersByTimeAsync(1500);

    // Wait for reconnection and data
    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
      expect(result.current.data).toEqual(mockData);
    });

    expect(connectionCount).toBeGreaterThanOrEqual(2);
    vi.useRealTimers();
  });

  test('useWebSocket does not connect when disabled', async () => {
    mockServer = new Server(WS_URL);
    const serverOnConnection = vi.fn();
    mockServer.on('connection', serverOnConnection);

    const { result } = renderHook(() => useWebSocket({ url: WS_URL, symbols: ['TEST'], enabled: false }));

    // Wait a moment to ensure no connection attempt
    await new Promise(resolve => setTimeout(resolve, 500));

    expect(result.current.isConnected).toBe(false);
    expect(serverOnConnection).not.toHaveBeenCalled();
  });
});
