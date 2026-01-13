import { renderHook, waitFor } from '@testing-library/react';
import { expect, test, beforeAll, afterEach, afterAll } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { usePortfolio } from '../src/features/portfolio/hooks/usePortfolio';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import React from 'react';

const handlers = [
  http.get('/api/v1/portfolio/summary', () => {
    return HttpResponse.json({
      totalValue: 125000.50,
      dailyPnL: 1200.25,
      dailyPnLPercent: 0.97,
      positionsCount: 12,
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

test('usePortfolio fetches portfolio summary', async () => {
  const { result } = renderHook(() => usePortfolio(), { wrapper: createWrapper() });

  await waitFor(() => expect(result.current.isSuccess).toBe(true));

  expect(result.current.data).toEqual({
    totalValue: 125000.50,
    dailyPnL: 1200.25,
    dailyPnLPercent: 0.97,
    positionsCount: 12,
  });
});