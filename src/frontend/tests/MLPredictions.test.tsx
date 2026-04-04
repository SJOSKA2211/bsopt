import { render, screen } from '@testing-library/react';
import { expect, test, vi } from 'vitest';
import { MLPredictions } from '../src/features/options/components/MLPredictions';
import { ThemeProvider } from '@mui/material/styles';
import { theme } from '../src/theme/index';
import '@testing-library/jest-dom';
import React from 'react';

// Mock Apollo hooks
vi.mock('@apollo/client', async (importOriginal) => {
  const actual = await importOriginal() as any;
  return {
    ...actual,
    useQuery: vi.fn(() => ({ data: undefined, loading: true, error: undefined, refetch: vi.fn() })),
    useSubscription: vi.fn(() => ({ data: undefined, loading: true, error: undefined })),
    useMutation: vi.fn(() => [vi.fn(), { data: undefined, loading: false, error: undefined }]),
    gql: actual.gql || ((strings: any) => strings[0]),
  };
});

import { useQuery } from '@apollo/client';

const mockPrediction = {
  mlPrediction: {
    symbol: 'AAPL',
    predictedPrice: 155.20,
    confidenceInterval: [153.50, 157.00],
    drift: 0.02,
    modelName: 'XGBoost-V4-Optimized',
    lastUpdated: '2026-03-19T00:00:00Z',
  },
};

test('MLPredictions renders prediction data correctly', async () => {
  (useQuery as any).mockReturnValue({
    data: mockPrediction,
    loading: false,
    error: null,
  });

  render(
    <ThemeProvider theme={theme}>
      <MLPredictions symbol="AAPL" />
    </ThemeProvider>
  );

  expect(await screen.findByText(/\$155\.20/)).toBeInTheDocument();
  expect(screen.getByText(/XGBoost-V4-Optimized/i)).toBeInTheDocument();
  expect(screen.getByText(/\+2\.00%/)).toBeInTheDocument();
});