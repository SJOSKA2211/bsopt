import { render, screen, waitFor } from '@testing-library/react';
import { expect, test, vi } from 'vitest';
import { LivePriceChart } from '../src/features/charts/components/LivePriceChart';
import { ThemeProvider } from '@mui/material/styles';
import { theme } from '../src/theme/index';
import { useWebSocket } from '../src/hooks/useWebSocket';
import React from 'react';

// Mock dependencies
vi.mock('../src/hooks/useWebSocket');

// Mock lightweight-charts
const mockUpdate = vi.fn();
const mockSetData = vi.fn();

vi.mock('lightweight-charts', () => ({
  createChart: vi.fn(() => ({
    addLineSeries: vi.fn(() => ({
      setData: mockSetData,
      update: mockUpdate,
    })),
    addCandlestickSeries: vi.fn(() => ({
      setData: mockSetData,
      update: mockUpdate,
    })),
    addSeries: vi.fn(() => ({
      setData: mockSetData,
      update: mockUpdate,
    })),
    applyOptions: vi.fn(),
    timeScale: vi.fn(() => ({
      fitContent: vi.fn(),
    })),
    remove: vi.fn(),
    resize: vi.fn(),
  })),
  ColorType: { Solid: 'solid' },
  CrosshairMode: { Normal: 0 },
  CandlestickSeries: 'CandlestickSeries',
  LineSeries: 'LineSeries',
}));

test('LivePriceChart renders chart container', () => {
  vi.mocked(useWebSocket).mockReturnValue({ data: null, isConnected: false });

  render(
    <ThemeProvider theme={theme}>
      <LivePriceChart symbol="AAPL" />
    </ThemeProvider>
  );

  expect(screen.getByTestId('live-price-chart-container')).toBeInTheDocument();
});

test('LivePriceChart updates with websocket data', async () => {
  const mockCandle = {
    symbol: 'AAPL',
    time: 1768226400,
    open: 150,
    high: 155,
    low: 149,
    close: 153
  };

  // Initially no data
  vi.mocked(useWebSocket).mockReturnValue({ data: null, isConnected: true });

  const { rerender } = render(
    <ThemeProvider theme={theme}>
      <LivePriceChart symbol="AAPL" />
    </ThemeProvider>
  );

  // Simulate data arrival
  vi.mocked(useWebSocket).mockReturnValue({ data: mockCandle, isConnected: true });

  rerender(
    <ThemeProvider theme={theme}>
      <LivePriceChart symbol="AAPL" />
    </ThemeProvider>
  );

  await waitFor(() => {
    expect(mockUpdate).toHaveBeenCalledWith({
      time: mockCandle.time,
      open: mockCandle.open,
      high: mockCandle.high,
      low: mockCandle.low,
      close: mockCandle.close
    });
  });
});