import { useQuery, useSubscription } from '@apollo/client/react';
import { gql } from '@apollo/client';

import type { PortfolioData } from '../types';

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
        entryPrice: entry_price
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


const authClient = { 
  signIn: { 
    social: async () => ({}) 
  }, 
  useSession: () => ({ 
    data: { 
      user: { 
        id: 'mock-user-123', 
        email: 'trader@bsopt.io', 
        name: 'Quant Trader' 
      } 
    },
    isLoading: false
  }) 
} as any;


const authClient = { 
  signIn: { 
    social: async () => ({}) 
  }, 
  useSession: () => ({ 
    data: { 
      user: { 
        id: 'mock-user-123', 
        email: 'trader@bsopt.io', 
        name: 'Quant Trader' 
      } 
    },
    isLoading: false
  }) 
} as any;

export const usePortfolio = () => {
  const { data: sessionData } = authClient.useSession();
  const userId = sessionData?.user?.id;

  // Initial fetch and polling fallback
  const { data, loading, error, refetch } = useQuery(GET_PORTFOLIO, {
    variables: { userId },
    skip: !userId,
    pollInterval: 10000, // Increased poll interval since we have subscriptions
  });

  const portfolioId = (data as { portfolio?: { id: string } })?.portfolio?.id;

  // Real-time updates via WebSocket
  useSubscription(PORTFOLIO_UPDATES, {
    variables: { portfolioId },
    skip: !portfolioId,
    onData: ({ data: subData }) => {
      // Apollo Cache will automatically merge this if the __typename and id match
      console.log('Portfolio real-time update received:', (subData.data as { portfolioUpdates?: unknown })?.portfolioUpdates);
    }
  });

  return {
    data: (data as { portfolio?: PortfolioData })?.portfolio,
    isLoading: loading,
    isError: !!error,
    error,
    refetch,
  };
};
