import { renderHook } from '@testing-library/react';
import { expect, test, vi } from 'vitest';
import { usePortfolio } from '../src/features/portfolio/hooks/usePortfolio';
import React from 'react';

// Mock auth client
vi.mock('../src/lib/auth-client', () => ({
  authClient: {
    useSession: () => ({ data: { user: { id: 'test-user-id' } } })
  }
}));

// Mock Apollo hooks from the correct entry point
vi.mock('@apollo/client/react', async (importOriginal) => {
  const actual = await importOriginal() as any;
  return {
    ...actual,
    useQuery: vi.fn(),
    useSubscription: vi.fn(),
  };
});

import { useQuery, useSubscription } from '@apollo/client/react';

test('usePortfolio fetches portfolio summary', async () => {
  const mockData = {
    portfolio: {
      id: 'port-1',
      balance: 100000,
      frozen_capital: 25000,
      risk_score: 0.15,
      totalValue: 125000.50,
      dailyPnL: 1200.25,
      dailyPnLPercent: 0.97,
      positionsCount: 12,
      positions: [
        {
          id: 'pos-1',
          contract_symbol: 'AAPL',
          quantity: 10,
          entryPrice: 150.0
        }
      ],
    }
  };

  (useQuery as any).mockReturnValue({
    data: { portfolio: mockData.portfolio },
    loading: false,
    error: null,
    refetch: vi.fn(),
  });

  (useSubscription as any).mockReturnValue({
    data: null,
    loading: false,
  });

  const { result } = renderHook(() => usePortfolio());

  expect(result.current.data).toBeDefined();
  expect(result.current.data?.totalValue).toEqual(125000.50);
  expect(result.current.data?.positions[0].contract_symbol).toEqual('AAPL');
});
