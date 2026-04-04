import { render, screen } from '@testing-library/react';
import { expect, test, vi } from 'vitest';
import { GreeksHeatmap } from '../src/features/options/components/GreeksHeatmap';
import { ThemeProvider } from '@mui/material/styles';
import { theme } from '../src/theme/index';
import React from 'react';

// Mock Apollo hooks
vi.mock('@apollo/client/react', async (importOriginal) => {
  const actual = await importOriginal() as any;
  return {
    ...actual,
    useQuery: vi.fn(),
  };
});

import { useQuery } from '@apollo/client/react';

// Mock useWasmPricing
vi.mock('../src/hooks/useWasmPricing', () => ({
  useWasmPricing: () => ({
    isLoaded: true,
    batchCalculate: vi.fn().mockResolvedValue([
      { greeks: { delta: 0.5, gamma: 0.05, vega: 0.1, theta: -0.01 } }
    ])
  })
}));

// Mock echarts-for-react
vi.mock('echarts-for-react', () => ({
  __esModule: true,
  default: () => <div data-testid="echarts-mock" />
}));

test('GreeksHeatmap renders container and chart after loading', async () => {
  const mockData = {
    marketData: { lastPrice: 155.0 },
    options: {
      edges: [
        {
          node: {
            id: 'opt1',
            strike: 150,
            expiry: '2026-01-20',
            optionType: 'CALL',
            iv: 0.2,
            delta: 0.5,
            gamma: 0.05,
          }
        }
      ]
    }
  };

  (useQuery as any).mockReturnValue({
    data: mockData,
    loading: false,
    error: null,
  });

  render(
    <ThemeProvider theme={theme}>
      <GreeksHeatmap symbol="AAPL" greek="delta" />
    </ThemeProvider>
  );

  // Should show container eventually
  expect(await screen.findByTestId('greeks-heatmap-container')).toBeInTheDocument();
  expect(screen.getByTestId('echarts-mock')).toBeInTheDocument();
});
