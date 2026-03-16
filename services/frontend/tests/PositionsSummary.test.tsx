import { render, screen } from '@testing-library/react';
import { expect, test, vi } from 'vitest';
import { PositionsSummary } from '../src/features/portfolio/components/PositionsSummary';
import { ThemeProvider } from '@mui/material/styles';
import { theme } from '../src/theme/index';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { usePortfolio } from '../src/features/portfolio/hooks/usePortfolio';
import React from 'react';

// Mock the hook
vi.mock('../src/features/portfolio/hooks/usePortfolio');

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
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

test('PositionsSummary renders portfolio data', async () => {
  const mockData = {
    balance: 123456.78,
    frozen_capital: 12345.67,
    risk_score: 0.25,
    totalValue: 135802.45,
    dailyPnL: 1000,
    dailyPnLPercent: 0.74,
    positionsCount: 2,
    positions: [
      { contract_symbol: 'AAPL_20260301_C_150', quantity: 5, current_pnl: 500.50 },
      { contract_symbol: 'TSLA_20260301_P_200', quantity: -2, current_pnl: -120.00 },
    ],
  };

  vi.mocked(usePortfolio).mockReturnValue({
    data: mockData,
    isLoading: false,
    isSuccess: true,
  } as any); // eslint-disable-line @typescript-eslint/no-explicit-any

  render(<PositionsSummary />, { wrapper: createWrapper() });

  expect(screen.getByText(/\$123,456.78/)).toBeInTheDocument();
  expect(screen.getByText(/\$12,345.67/)).toBeInTheDocument();
  expect(screen.getByText(/0.25/)).toBeInTheDocument();
  
  expect(screen.getByText('AAPL_20260301_C_150')).toBeInTheDocument();
  expect(screen.getByText('TSLA_20260301_P_200')).toBeInTheDocument();
  expect(screen.getByText(/\$500.50/)).toBeInTheDocument();
  expect(screen.getByText(/-\$120.00/)).toBeInTheDocument();
});

test('PositionsSummary shows loading state', () => {
  vi.mocked(usePortfolio).mockReturnValue({
    isLoading: true,
  } as any); // eslint-disable-line @typescript-eslint/no-explicit-any

  render(<PositionsSummary />, { wrapper: createWrapper() });
  expect(screen.getByRole('progressbar')).toBeInTheDocument();
});
