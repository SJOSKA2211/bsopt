import { useQuery } from '@tanstack/react-query';
import type { PortfolioData } from '../types';

export const usePortfolio = () => {
  return useQuery<PortfolioData>({
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
