import { render, screen, waitFor } from '@testing-library/react';
import { expect, test, vi } from 'vitest';
import { OptionsChain } from '../src/features/options/components/OptionsChain';
import { ThemeProvider } from '@mui/material/styles';
import { theme } from '../src/theme/index';
import React from 'react';

// Mock Apollo hooks
vi.mock('@apollo/client/react', async (importOriginal) => {
  const actual = await importOriginal() as any;
  return {
    ...actual,
    useQuery: vi.fn(),
    useSubscription: vi.fn(),
  };
});

import { useQuery } from '@apollo/client/react';

// Mock useWasmPricing
vi.mock('../src/hooks/useWasmPricing', () => ({
  useWasmPricing: () => ({
    isLoaded: true,
    batchCalculate: vi.fn().mockResolvedValue([])
  })
}));

// Mock data for the test
const mockData = {
  marketData: { lastPrice: 100.50 },
  options: {
    edges: [
      {
        node: {
          id: 'call-1', strike: 100, expiry: '2026-03-01', optionType: 'call',
          bid: 1.50, ask: 1.60, lastPrice: 1.55, volume: 100, openInterest: 500, iv: 0.20, price: 1.55, delta: 0.55, gamma: 0.05
        }
      }
    ]
  }
};

const createWrapper = () => {
  return ({ children }: { children: React.ReactNode }) => (
    <ThemeProvider theme={theme}>
      {children}
    </ThemeProvider>
  );
};

test('OptionsChain fetches and displays data', async () => {
  (useQuery as any).mockReturnValue({
    data: mockData,
    loading: false,
    error: null,
  });

  render(<OptionsChain symbol="AAPL" />, { wrapper: createWrapper() });

  // Verify accessibility labels
  expect(screen.getByLabelText('Search by strike price')).toBeInTheDocument();
  expect(screen.getByLabelText('Select pricing model')).toBeInTheDocument();
  expect(screen.getByLabelText('Filter by expiration')).toBeInTheDocument();

  // Wait for the mock data to appear
  await waitFor(() => {
    expect(screen.getByText('Options Chain - AAPL')).toBeInTheDocument();
    expect(screen.getByText('$1.50')).toBeInTheDocument(); // call_bid
    expect(screen.getByText('100')).toBeInTheDocument(); // strike

    // Verify Greeks cell button accessibility (might be multiple)
    const greeksButtons = screen.queryAllByLabelText(/View Greeks details|Greeks calculation pending/);
    expect(greeksButtons.length).toBeGreaterThan(0);
  }, { timeout: 2000 });
});