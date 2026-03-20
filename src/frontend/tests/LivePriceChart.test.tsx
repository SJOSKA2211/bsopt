import { render, screen } from '@testing-library/react';
import { expect, test } from 'vitest';
import { LivePriceChart } from '../src/features/charts/components/LivePriceChart';
import { ThemeProvider } from '@mui/material/styles';
import { theme } from '../src/theme/index';
import { MockedProvider } from '@apollo/client/testing/react';
import '@testing-library/jest-dom';
import { gql } from '@apollo/client';
import React from 'react';

const MARKET_SUBSCRIPTION = gql`
  subscription OnMarketUpdate($symbols: [String!]!) {
    market_data_stream(symbols: $symbols) {
      symbol
      lastPrice: last_price
      volume
    }
  }
`;

const GET_HISTORICAL_DATA = gql`
  query GetHistoricalData($symbol: String!) {
    historicalData(symbol: $symbol) {
      time
      open
      high
      low
      close
    }
  }
`;

const mocks = [
  {
    request: {
      query: GET_HISTORICAL_DATA,
      variables: { symbol: 'AAPL' },
    },
    result: {
      data: {
        historicalData: [
          { time: 1768226400, open: 150, high: 155, low: 145, close: 152 },
        ],
      },
    },
  },
  {
    request: {
      query: MARKET_SUBSCRIPTION,
      variables: { symbols: ['AAPL'] },
    },
    result: {
      data: {
        market_data_stream: {
          symbol: 'AAPL',
          lastPrice: 153.50,
          volume: 1000,
        },
      },
    },
  },
];

test('LivePriceChart renders chart container', () => {
  render(
    <MockedProvider mocks={mocks}>
      <ThemeProvider theme={theme}>
        <LivePriceChart symbol="AAPL" />
      </ThemeProvider>
    </MockedProvider>
  );

  expect(screen.getByTestId('live-price-chart-container')).toBeInTheDocument();
});