import { render, screen } from '@testing-library/react';
import { expect, test, vi } from 'vitest';
import { PortfolioSummary } from '../src/features/portfolio/components/PortfolioSummary';
import { ThemeProvider } from '@mui/material/styles';
import { theme } from '../src/theme/index';
import React from 'react';

// Mock usePortfolio hook
vi.mock('../src/features/portfolio/hooks/usePortfolio', () => ({
  usePortfolio: vi.fn(),
}));

import { usePortfolio } from '../src/features/portfolio/hooks/usePortfolio';

// Mock data for the test
const mockPortfolioData = {
  id: 'port-1',
  balance: 100000,
  frozen_capital: 20000,
  risk_score: 0.1,
  totalValue: 125000.50,
  dailyPnL: 1200.25,
  dailyPnLPercent: 0.97,
  positionsCount: 12,
  positions: [],
};

const createWrapper = () => {
  return ({ children }: { children: React.ReactNode }) => (
    <ThemeProvider theme={theme}>
      {children}
    </ThemeProvider>
  );
};

test('PortfolioSummary displays values correctly', async () => {
  (usePortfolio as any).mockReturnValue({
    data: mockPortfolioData,
    isLoading: false,
    isError: false,
  });

  render(<PortfolioSummary />, { wrapper: createWrapper() });

  expect(await screen.findByText(/125,000\.50/)).toBeInTheDocument();
  expect(await screen.findByText(/1,200\.25/)).toBeInTheDocument();
  expect(await screen.findByText(/0\.97%/)).toBeInTheDocument();
  expect(await screen.findByText(/12\s*UNITS/)).toBeInTheDocument();
});