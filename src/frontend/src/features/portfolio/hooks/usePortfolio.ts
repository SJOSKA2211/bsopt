import { useQuery } from '@tanstack/react-query';

export interface PortfolioSummary {
  totalValue: number;
  dailyPnL: number;
  dailyPnLPercent: number;
  positionsCount: number;
}

export const usePortfolio = () => {
  return useQuery<PortfolioSummary>({
    queryKey: ['portfolio-summary'],
    queryFn: async () => {
      const response = await fetch('/api/v1/portfolio/summary');
      if (!response.ok) {
        throw new Error('Failed to fetch portfolio summary');
      }
      return response.json();
    },
    refetchInterval: 5000, // Refresh every 5 seconds
  });
};