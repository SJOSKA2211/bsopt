import { renderHook, waitFor } from '@testing-library/react';
import { expect, test } from 'vitest';
import { MockedProvider } from '@apollo/client/testing/react';
import { usePortfolio } from '../src/features/portfolio/hooks/usePortfolio';
import { gql } from '@apollo/client';
import React from 'react';
import { vi } from 'vitest';

// Mock auth client
vi.mock('../src/lib/auth-client', () => ({
  authClient: {
    useSession: () => ({ data: { user: { id: 'test-user-id' } } })
  }
}));

const GET_PORTFOLIO = gql`
  query GetPortfolio($userId: String!) {
    portfolio(userId: $userId) {
      id
      balance
      frozen_capital
      risk_score
      totalValue: total_value
      dailyPnL: daily_pnl
      dailyPnLPercent: daily_pnl_percent
      positionsCount: positions_count
      positions {
        id
        contract_symbol
        quantity
        entry_price
      }
    }
  }
`;

const PORTFOLIO_UPDATES = gql`
  subscription OnPortfolioUpdate($portfolioId: ID!) {
    portfolioUpdates(portfolioId: $portfolioId) {
      id
      balance
      frozen_capital
      risk_score
      total_value
      daily_pnl
      daily_pnl_percent
      positions_count
    }
  }
`;

const mocks = [
  {
    request: {
      query: GET_PORTFOLIO,
      variables: { userId: 'test-user-id' },
    },
    result: {
      data: {
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
              entry_price: 150.0
            }
          ],
        }
      }
    }
  },
  {
    request: {
      query: PORTFOLIO_UPDATES,
      variables: { portfolioId: 'port-1' },
    },
    result: {
      data: {
        portfolioUpdates: {
          id: 'port-1',
          balance: 100000,
          frozen_capital: 25000,
          risk_score: 0.15,
          total_value: 125000.50,
          daily_pnl: 1200.25,
          daily_pnl_percent: 0.97,
          positions_count: 12,
        }
      }
    }
  }
];

const createWrapper = () => {
  return ({ children }: { children: React.ReactNode }) => (
    <MockedProvider mocks={mocks}>
      {children}
    </MockedProvider>
  );
};

test('usePortfolio fetches portfolio summary', async () => {
  const { result } = renderHook(() => usePortfolio(), { wrapper: createWrapper() });

  await waitFor(() => expect(result.current.data).toBeDefined(), { timeout: 2000 });

  expect(result.current.data?.totalValue).toEqual(125000.50);
});
