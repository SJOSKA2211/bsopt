import { render, screen } from '@testing-library/react';
import { expect, test } from 'vitest';
import { TradingDashboard } from '../src/features/dashboard/components/TradingDashboard';
import { ThemeProvider } from '@mui/material/styles';
import { theme } from '../src/theme/index';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';

const createWrapper = () => {
  const queryClient = new QueryClient();
  return ({ children }: { children: React.ReactNode }) => (
    <ThemeProvider theme={theme}>
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    </ThemeProvider>
  );
};

test('TradingDashboard renders layout components', () => {
  render(<TradingDashboard />, { wrapper: createWrapper() });

  expect(screen.getByText(/BS-Opt Trading Dashboard/i)).toBeInTheDocument();
  expect(screen.getByRole('navigation')).toBeInTheDocument();
  expect(screen.getByTestId('options-chain-container')).toBeInTheDocument();
  expect(screen.getByTestId('portfolio-summary-container')).toBeInTheDocument();
});
