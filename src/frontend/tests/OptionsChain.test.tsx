import { render, screen, waitFor } from '@testing-library/react';
import { expect, test, beforeAll, afterEach, afterAll } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { OptionsChain } from '../src/features/options/components/OptionsChain';
import { ThemeProvider } from '@mui/material/styles';
import { theme } from '../src/theme/index';
import { MockedProvider } from '@apollo/client/testing';
import { gql } from '@apollo/client';
import React from 'react';

const GET_OPTIONS_CHAIN = gql`
  query GetOptionsChain($symbol: String!, $expiryBucket: String) {
    marketData(symbol: $symbol) {
      lastPrice
    }
    options(underlying: $symbol, expiryBucket: $expiryBucket) {
      edges {
        node {
          id
          strike
          expiry
          optionType
          bid
          ask
          lastPrice
          volume
          openInterest
          iv
          price
          delta
          gamma
        }
      }
    }
  }
`;

const mocks = [
  {
    request: {
      query: GET_OPTIONS_CHAIN,
      variables: { symbol: 'AAPL', expiryBucket: 'all' },
    },
    result: {
      data: {
        marketData: {
          lastPrice: 100.50,
        },
        options: {
          edges: [
            {
              node: {
                id: 'call-1', strike: 100, expiry: '2026-03-01', optionType: 'call',
                bid: 1.50, ask: 1.60, lastPrice: 1.55, volume: 100, openInterest: 500, iv: 0.20, price: 1.55, delta: 0.55, gamma: 0.05
              }
            },
            {
              node: {
                id: 'put-1', strike: 100, expiry: '2026-03-01', optionType: 'put',
                bid: 0.50, ask: 0.60, lastPrice: 0.55, volume: 80, openInterest: 400, iv: 0.22, price: 0.55, delta: -0.45, gamma: 0.04
              }
            }
          ]
        }
      }
    }
  }
];

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false, // Disable retries for tests
      },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <ThemeProvider theme={theme}>
      <QueryClientProvider client={queryClient}>
        <MockedProvider mocks={mocks} addTypename={false}>
          {children}
        </MockedProvider>
      </QueryClientProvider>
    </ThemeProvider>
  );
};

test('OptionsChain fetches and displays data', async () => {
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