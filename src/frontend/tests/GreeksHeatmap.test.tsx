import { render, screen } from '@testing-library/react';
import { expect, test, vi, beforeAll, afterEach, afterAll } from 'vitest';
import { GreeksHeatmap } from '../src/features/options/components/GreeksHeatmap';
import { ThemeProvider } from '@mui/material/styles';
import { theme } from '../src/theme/index';
import { MockedProvider } from '@apollo/client/testing/react';
import { gql } from '@apollo/client';
import React from 'react';

const GET_OPTIONS_FOR_HEATMAP = gql`
  query GetOptionsForHeatmap($symbol: String!) {
    marketData(symbol: $symbol) {
      lastPrice
    }
    options(underlying: $symbol) {
      edges {
        node {
          id
          strike
          expiry
          optionType
          iv
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
      query: GET_OPTIONS_FOR_HEATMAP,
      variables: { symbol: 'AAPL' },
    },
    result: {
      data: {
        marketData: { lastPrice: 155.0 },
        options: {
          edges: [
            {
              node: {
                id: 'opt1',
                strike: 150,
                expiry: '2026-01-20',
                optionType: 'CALL',
                iv: 0.2,
                delta: 0.5,
                gamma: 0.05,
              }
            }
          ]
        }
      }
    }
  }
];

// Remove MSW setup as we'll use MockedProvider

// Mock useWasmPricing
vi.mock('../src/hooks/useWasmPricing', () => ({
  useWasmPricing: () => ({
    isLoaded: true,
    batchCalculate: vi.fn().mockResolvedValue([
      { greeks: { delta: 0.5, gamma: 0.05, vega: 0.1, theta: -0.01 } }
    ])
  })
}));

// Mock echarts-for-react
vi.mock('echarts-for-react/lib/src.shared', () => ({
  default: () => <div data-testid="echarts-mock" />
}));

const createWrapper = () => {
  return ({ children }: { children: React.ReactNode }) => (
    <MockedProvider mocks={mocks}>
      <ThemeProvider theme={theme}>
        {children}
      </ThemeProvider>
    </MockedProvider>
  );
};

test('GreeksHeatmap renders container and chart after loading', async () => {
  render(<GreeksHeatmap symbol="AAPL" greek="delta" />, { wrapper: createWrapper() });

  // Should show container eventually
  expect(await screen.findByTestId('greeks-heatmap-container')).toBeInTheDocument();
  expect(screen.getByTestId('echarts-mock')).toBeInTheDocument();
});
