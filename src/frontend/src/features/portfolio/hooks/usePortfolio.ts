import { useQuery, gql } from '@apollo/client';
import { authClient } from '../../../lib/auth-client';
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

export const usePortfolio = () => {
  const { data: sessionData } = authClient.useSession();
  const userId = sessionData?.user?.id || "user_123";

  const { data, loading, error, refetch } = useQuery(GET_PORTFOLIO, {
    variables: { userId },
    pollInterval: 5000,
  });

  return {
    data: data?.portfolio as PortfolioData | undefined,
    isLoading: loading,
    isError: !!error,
    error,
    refetch,
  };
};
