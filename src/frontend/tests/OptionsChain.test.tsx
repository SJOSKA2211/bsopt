import { render, screen, waitFor } from '@testing-library/react';
import { expect, test, beforeAll, afterEach, afterAll } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { OptionsChain } from '../src/features/options/components/OptionsChain';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { ThemeProvider } from '@mui/material/styles';
import { theme } from '../src/theme/index';
import React from 'react';

// Mock Server Setup
const handlers = [
  http.get('/api/v1/options/chain', () => {
    // Basic mock response
    return HttpResponse.json([
      {
        id: 'mock-1', strike: 100, expiry: '2026-03-01', underlying_price: 100.50,
        call_bid: 1.50, call_ask: 1.60, call_last: 1.55, call_volume: 100, call_oi: 500, call_iv: 0.20, call_delta: 0.55, call_gamma: 0.05,
        put_bid: 0.50, put_ask: 0.60, put_last: 0.55, put_volume: 80, put_oi: 400, put_iv: 0.22, put_delta: -0.45, put_gamma: 0.04,
      },
    ]);
  }),
];

const server = setupServer(...handlers);

// Establish API mocking before all tests.
beforeAll(() => server.listen());
// Reset any request handlers that are declared as a part of our tests
// (i.e. for testing one-off error cases).
afterEach(() => server.resetHandlers());
// Clean up after the tests are finished.
afterAll(() => server.close());


// Test component wrapper with QueryClientProvider and ThemeProvider
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false, // Disable retries for tests
      },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <ThemeProvider theme={theme}>
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    </ThemeProvider>
  );
};

test('OptionsChain fetches and displays data', async () => {
  render(<OptionsChain symbol="AAPL" />, { wrapper: createWrapper() });

  // Expect a loading state initially (optional, but good practice)
  // expect(screen.getByText(/loading/i)).toBeInTheDocument();

  // Wait for the mock data to appear
  await waitFor(() => {
    expect(screen.getByText('Options Chain - AAPL')).toBeInTheDocument();
    expect(screen.getByText('$1.50')).toBeInTheDocument(); // call_bid
    expect(screen.getByText('100')).toBeInTheDocument(); // strike
  }, { timeout: 2000 });
});