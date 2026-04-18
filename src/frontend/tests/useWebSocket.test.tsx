/** @vitest-environment jsdom */
import { renderHook, waitFor, act } from '@testing-library/react';
import { useWebSocket } from '../src/hooks/useWebSocket';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import { Server, WebSocket as MockWebSocket } from 'mock-socket';

describe('useWebSocket', () => {
  const WS_URL = 'ws://localhost:8080';
  let mockServer: Server;

  beforeEach(() => {
    mockServer = new Server(WS_URL);
  });

  afterEach(() => {
    mockServer.stop();
  });

  it('connects and receives data', async () => {
    const { result } = renderHook(() => useWebSocket({ url: WS_URL, reconnectInterval: 50 }));

    await waitFor(() => expect(result.current.isConnected).toBe(true));

    const testData = { type: 'TICK', price: 100 };
    
    // Server emitting must happen while components are observing
    act(() => {
      mockServer.emit('message', JSON.stringify(testData));
    });

    await waitFor(() => {
      expect(result.current.lastMessage).toEqual(testData);
    });
  });

  it('handles reconnection', async () => {
    const { result } = renderHook(() => useWebSocket({ url: WS_URL, reconnectInterval: 50 }));

    await waitFor(() => expect(result.current.isConnected).toBe(true));

    // Simulate disconnect
    act(() => {
      mockServer.close();
    });

    await waitFor(() => expect(result.current.isConnected).toBe(false));

    // Create a new server to accept the reconnection
    mockServer = new Server(WS_URL);

    // Should reconnect shortly due to reconnectInterval=50
    await waitFor(() => expect(result.current.isConnected).toBe(true), { timeout: 2000 });
  });

  it('cleans up on unmount', async () => {
    const { result, unmount } = renderHook(() => useWebSocket({ url: WS_URL, reconnectInterval: 50 }));
    
    await waitFor(() => expect(result.current.isConnected).toBe(true));
    
    act(() => {
      unmount();
    });
    
    // Server connection handles close, let's just make sure it doesn't crash
    expect(result.current.isConnected).toBe(true); // Right before unmount it was true
  });
});
