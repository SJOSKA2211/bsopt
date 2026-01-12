import { renderHook, act, waitFor } from '@testing-library/react';
import { Server } from 'mock-socket';
import { useWebSocket } from '../src/hooks/useWebSocket';
import { vi, expect, test } from 'vitest';

const WS_URL = 'ws://localhost:1234';

test('useWebSocket connects and receives data', async () => {
  const mockServer = new Server(WS_URL);
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
  }, { timeout: 2000 }); // Increase timeout for WebSocket connection

  mockServer.stop();
});

test('useWebSocket handles reconnection', async () => {
  const mockServer = new Server(WS_URL);
  const mockData = { price: 101, symbol: 'TEST' };
  
  let connectionCount = 0;
  mockServer.on('connection', socket => {
    connectionCount++;
    if (connectionCount === 1) {
      // Close first connection to simulate disconnect
      socket.close();
    } else {
      socket.send(JSON.stringify(mockData));
    }
  });

  const { result } = renderHook(() => useWebSocket({ url: WS_URL, symbols: ['TEST'], enabled: true }));

  // Wait for initial connection, then disconnection, then reconnection and data
  await waitFor(() => {
    expect(result.current.isConnected).toBe(true); // First connection
  }, { timeout: 2000 });

  await waitFor(() => {
    expect(result.current.isConnected).toBe(false); // Disconnected
  }, { timeout: 2000 });

  await waitFor(() => {
    expect(result.current.isConnected).toBe(true); // Reconnected
    expect(result.current.data).toEqual(mockData);
  }, { timeout: 4000 }); // Longer timeout for reconnection

  expect(connectionCount).toBeGreaterThanOrEqual(2); // Should have connected at least twice
  mockServer.stop();
});

test('useWebSocket does not connect when disabled', async () => {
  const mockServer = new Server(WS_URL);
  const serverOnConnection = vi.fn();
  mockServer.on('connection', serverOnConnection);

  const { result } = renderHook(() => useWebSocket({ url: WS_URL, symbols: ['TEST'], enabled: false }));

  // Wait a moment to ensure no connection attempt
  await new Promise(resolve => setTimeout(resolve, 500));

  expect(result.current.isConnected).toBe(false);
  expect(serverOnConnection).not.toHaveBeenCalled();

  mockServer.stop();
});
