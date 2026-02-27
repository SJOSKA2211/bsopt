import { render, screen, waitFor } from '@testing-library/react';
import { expect, test, beforeAll, afterEach, afterAll } from 'vitest';
import { DashboardPage } from '../src/pages/dashboard/DashboardPage';
import { Layout } from '../src/components/layout/Layout';
import { ThemeProvider } from '@mui/material/styles';
import { theme } from '../src/theme/index';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import React from 'react';
import { BrowserRouter } from 'react-router-dom';

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
  const queryClient = new QueryClient();
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

  expect(screen.getByText(/CASHMATE/i)).toBeInTheDocument();
  
  // Wait for DashboardPage to render (not just the layout)
  await waitFor(() => {
    expect(screen.getByText(/TOTAL SPENDINGS/i)).toBeInTheDocument();
  }, { timeout: 10000 });

  expect(screen.getByTestId('options-chain-container')).toBeInTheDocument();
  expect(screen.getByTestId('portfolio-summary-container')).toBeInTheDocument();
  expect(screen.getByTestId('live-price-chart-paper')).toBeInTheDocument();
  expect(screen.getByTestId('greeks-heatmap-paper')).toBeInTheDocument();
  expect(screen.getByTestId('ml-predictions-paper')).toBeInTheDocument();
  expect(screen.getByTestId('volatility-surface-paper')).toBeInTheDocument();
  
  // Wait for portfolio summary to load
  expect(await screen.findByText(/Portfolio Overview/i)).toBeInTheDocument();
});