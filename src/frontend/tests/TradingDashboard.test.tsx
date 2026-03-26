import { render, screen, waitFor } from '@testing-library/react';
import { expect, test, beforeAll, afterEach, afterAll, vi } from 'vitest';
import { DashboardPage } from '../src/pages/dashboard/DashboardPage';
import { Layout } from '../src/components/layout/Layout';
import { ThemeProvider } from '@mui/material/styles';
import { theme } from '../src/theme/index';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import React from 'react';
import { BrowserRouter } from 'react-router-dom';

// Mock heavy/complex components to isolate Dashboard rendering logic
vi.mock('../src/features/options/components/GreeksHeatmap', () => ({
  GreeksHeatmap: () => <div data-testid="greeks-heatmap-mock">Greeks Heatmap Mock</div>
}));

vi.mock('../src/features/options/components/VolatilitySurface3D', () => ({
  VolatilitySurface3D: () => <div data-testid="volatility-surface-mock">Volatility Surface Mock</div>
}));

// Mock LivePriceChart if needed (though lightweight-charts is mocked globally)
vi.mock('../src/features/charts/components/LivePriceChart', () => ({
  LivePriceChart: () => <div data-testid="live-price-chart-mock">Live Price Chart Mock</div>
}));

const handlers = [
  http.get('/api/v1/portfolio/summary', () => {
    return HttpResponse.json({
      balance: 100000,
      frozen_capital: 20000,
      risk_score: 0.1,
      totalValue: 125000.50,
      dailyPnL: 1200.25,
      dailyPnLPercent: 0.97,
      positionsCount: 12,
      positions: [],
    });
  }),
  http.get('/api/v1/options/chain', () => {
    return HttpResponse.json([]);
  }),
  http.get('/api/v1/ml/predictions', () => {
    return HttpResponse.json({
      symbol: 'AAPL',
      predictedPrice: 155.20,
      confidenceInterval: [153.50, 157.00],
      drift: 0.02,
      modelName: 'XGBoost-V4-Optimized',
      lastUpdated: new Date().toISOString(),
    });
  }),
];

const server = setupServer(...handlers);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <ThemeProvider theme={theme}>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          {children}
        </BrowserRouter>
      </QueryClientProvider>
    </ThemeProvider>
  );
};

test('Dashboard renders with Layout and trading components', async () => {
  render(
    <Layout>
      <DashboardPage />
    </Layout>, 
    { wrapper: createWrapper() }
  );

  expect(screen.getAllByText(/BS-Opt/i)[0]).toBeInTheDocument();
  
  // Wait for DashboardPage to render (not just the layout)
  await waitFor(() => {
    expect(screen.getAllByText(/Dashboard/i)[0]).toBeInTheDocument();
  }, { timeout: 10000 });
  
  // Wait for mocks to appear (proving they were loaded)
  expect(await screen.findByTestId('greeks-heatmap-mock')).toBeInTheDocument();
  expect(await screen.findByTestId('live-price-chart-mock')).toBeInTheDocument();
});
