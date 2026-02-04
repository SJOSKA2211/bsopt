import { render, screen } from '@testing-library/react';
import { expect, test, beforeAll, afterEach, afterAll } from 'vitest';
import { MLPredictions } from '../src/features/options/components/MLPredictions';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';

// Mock Server Setup
const handlers = [
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
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
};

test('MLPredictions renders prediction data correctly', async () => {
  render(<MLPredictions symbol="AAPL" />, { wrapper: createWrapper() });

  expect(await screen.findByText(/155\.20/)).toBeInTheDocument();
  expect(screen.getByText(/XGBoost-V4-Optimized/i)).toBeInTheDocument();
  expect(screen.getByText(/\+2\.00%/)).toBeInTheDocument(); // 0.02 drift
});