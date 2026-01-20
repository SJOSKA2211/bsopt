import { render, screen } from '@testing-library/react';
import { expect, test, vi, beforeAll, afterEach, afterAll } from 'vitest';
import { GreeksHeatmap } from '../src/features/options/components/GreeksHeatmap';
import { ThemeProvider } from '@mui/material/styles';
import { theme } from '../src/theme/index';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import React from 'react';

// Mock Server Setup
const handlers = [
  http.get('/api/v1/options/chain', () => {
    return HttpResponse.json([
      { strike: 150, expiry: '2026-01-20', call_delta: 0.5, call_gamma: 0.05, call_iv: 0.2, put_delta: -0.5, put_gamma: 0.05, put_iv: 0.2 },
    ]);
  }),
];

const server = setupServer(...handlers);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

// Mock echarts-for-react
vi.mock('echarts-for-react/lib/core', () => ({
  default: () => <div data-testid="echarts-mock" />
}));

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

test('GreeksHeatmap renders container and chart after loading', async () => {
  render(<GreeksHeatmap symbol="AAPL" greek="delta" />, { wrapper: createWrapper() });

  // Should show container eventually
  expect(await screen.findByTestId('greeks-heatmap-container')).toBeInTheDocument();
  expect(screen.getByTestId('echarts-mock')).toBeInTheDocument();
});
