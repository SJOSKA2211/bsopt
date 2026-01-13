import { render, screen } from '@testing-library/react';
import { expect, test, beforeAll, afterEach, afterAll } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PortfolioSummary } from '../src/features/portfolio/components/PortfolioSummary';
import { ThemeProvider } from '@mui/material/styles';
import { theme } from '../src/theme/index';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import React from 'react';

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
        {children}
      </QueryClientProvider>
    </ThemeProvider>
  );
};

test('PortfolioSummary displays values correctly', async () => {
  render(<PortfolioSummary />, { wrapper: createWrapper() });

  expect(await screen.findByText(/125,000\.50/)).toBeInTheDocument();
  expect(await screen.findByText(/1,200\.25/)).toBeInTheDocument();
  expect(await screen.findByText(/0\.97%/)).toBeInTheDocument();
  expect(await screen.findByText(/12 Positions/)).toBeInTheDocument();
});