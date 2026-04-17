import { render, screen } from '@testing-library/react';
import { expect, test, vi } from 'vitest';
import { LivePriceChart } from '../src/features/charts/components/LivePriceChart';
import { ThemeProvider } from '@mui/material/styles';
import { theme } from '../src/theme/index';
import '@testing-library/jest-dom';
import React from 'react';

// Mock Apollo hooks
vi.mock('@apollo/client/react', async (importOriginal) => {
  const actual = await importOriginal() as any;
  return {
    ...actual,
    useQuery: vi.fn(),
    useSubscription: vi.fn(),
  };
});

import { useQuery, useSubscription } from '@apollo/client/react';
import { useMotionValue } from 'framer-motion';

// Mock lightweight-charts
vi.mock('lightweight-charts', () => {
  const seriesInstance = {
    setData: vi.fn(),
    update: vi.fn(),
    applyOptions: vi.fn(),
    priceScale: vi.fn(() => ({
      applyOptions: vi.fn(),
    })),
  };
  
  return {
    createChart: vi.fn().mockReturnValue({
      addSeries: vi.fn().mockReturnValue(seriesInstance),
      addLineSeries: vi.fn().mockReturnValue(seriesInstance),
      addCandlestickSeries: vi.fn().mockReturnValue(seriesInstance),
      remove: vi.fn(),
      applyOptions: vi.fn(),
      subscribeCrosshairMove: vi.fn(),
      timeScale: vi.fn().mockReturnValue({ fitContent: vi.fn() }),
      resize: vi.fn(),
    }),
    ColorType: { Solid: 0 },
    CrosshairMode: { Normal: 0 },
    CandlestickSeries: 'CandlestickSeries',
    HistogramSeries: 'HistogramSeries',
    LineSeries: 'LineSeries',
  };
});


const mockHistoricalData = [
  { time: 1768226400, open: 150, high: 155, low: 145, close: 152 },
];

const mockUpdate = {
  symbol: 'AAPL',
  lastPrice: 153.50,
  volume: 1000,
};

test('LivePriceChart renders chart container', () => {
  (useQuery as any).mockReturnValue({
    data: { historicalData: mockHistoricalData },
    loading: false,
    error: null,
  });

  (useSubscription as any).mockReturnValue({
    data: { market_data_stream: mockUpdate },
    loading: false,
  });

  render(
    <ThemeProvider theme={theme}>
      <LivePriceChart symbol="AAPL" />
    </ThemeProvider>
  );

  expect(screen.getByTestId('live-price-chart-container')).toBeInTheDocument();
});