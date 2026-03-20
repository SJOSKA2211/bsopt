import { render, screen } from '@testing-library/react';
import { expect, test } from 'vitest';
import { MLPredictions, GET_ML_PREDICTION } from '../src/features/options/components/MLPredictions';
import { MockedProvider } from '@apollo/client/testing/react';
import '@testing-library/jest-dom';
import React from 'react';

const mocks = [
  {
    request: {
      query: GET_ML_PREDICTION,
      variables: { symbol: 'AAPL' },
    },
    result: {
      data: {
        mlPrediction: {
          symbol: 'AAPL',
          predictedPrice: 155.20,
          confidenceInterval: [153.50, 157.00],
          drift: 0.02,
          modelName: 'XGBoost-V4-Optimized',
          lastUpdated: '2026-03-19T00:00:00Z',
        },
      },
    },
  },
];

test('MLPredictions renders prediction data correctly', async () => {
  render(
    <MockedProvider mocks={mocks}>
      <MLPredictions symbol="AAPL" />
    </MockedProvider>
  );

  expect(await screen.findByText(/\$155\.20/)).toBeInTheDocument();
  expect(screen.getByText(/XGBoost-V4-Optimized/i)).toBeInTheDocument();
  expect(screen.getByText(/\+2\.00%/)).toBeInTheDocument();
});