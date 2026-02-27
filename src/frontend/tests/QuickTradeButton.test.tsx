import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { expect, test, beforeAll, afterEach, afterAll } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { QuickTradeButton } from '../src/features/options/components/QuickTradeButton';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import React from 'react';

const handlers = [
  http.post('/api/v1/trades/execute', () => {
    return HttpResponse.json({ success: true, message: 'Trade executed successfully', tradeId: 'TRD-123' }, { status: 200 });
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

const mockOption = {
  id: 'option-1',
  strike: 100,
  expiry: '2026-03-01',
  underlying_price: 100.50,
  call_bid: 1.50, call_ask: 1.60, call_last: 1.55, call_volume: 100, call_oi: 500, call_iv: 0.20, call_delta: 0.55, call_gamma: 0.05,
  put_bid: 0.50, put_ask: 0.60, put_last: 0.55, put_volume: 80, put_oi: 400, put_iv: 0.22, put_delta: -0.45, put_gamma: 0.04,
};

test('QuickTradeButton shows confirmation dialog and executes trade', async () => {
  render(
    <QuickTradeButton option={mockOption} type="call" action="buy" />,
    { wrapper: createWrapper() }
  );

  const buyButton = screen.getByRole('button', { name: /buy/i });
  fireEvent.click(buyButton);

  // Expect confirmation dialog
  expect(screen.getByText(/Confirm Trade/i)).toBeInTheDocument();
  expect(screen.getByText(/Are you sure you want to buy call option/i)).toBeInTheDocument();

  // Click Confirm
  const confirmButton = screen.getByRole('button', { name: /confirm/i });
  fireEvent.click(confirmButton);

  // Expect success message
  await waitFor(() => {
    expect(screen.getByText(/Trade executed successfully/i)).toBeInTheDocument();
  });
});