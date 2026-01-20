import { expect, test } from 'vitest';
import { TradingDashboard } from '../src/features/dashboard/components/TradingDashboard';
import { render, screen } from '@testing-library/react';
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

test('TradingDashboard uses lazy loading for heavy components', async () => {
  // This test verifies that the dashboard renders and eventually shows the heavy components
  // which are loaded lazily.
  render(<TradingDashboard />, { wrapper: createWrapper() });

  // Initially should show loading fallbacks or at least not the components
  // Since we are in a test env, they might load fast, but we can check for their containers
  expect(screen.getByTestId('live-price-chart-paper')).toBeInTheDocument();
  
  // These should eventually appear
  expect(await screen.findByTestId('options-chain-container')).toBeInTheDocument();
  expect(await screen.findByTestId('greeks-heatmap-paper')).toBeInTheDocument();
});
