import { render, screen } from '@testing-library/react';
import { expect, test, vi } from 'vitest';
import { MLPredictions } from '../src/features/options/components/MLPredictions';
import { ThemeProvider } from '@mui/material/styles';
import { theme } from '../src/theme/index';
import '@testing-library/jest-dom';
import React from 'react';

// Mock useMLInference
vi.mock('../src/api/hooks', () => ({
  useMLInference: vi.fn(),
}));

import { useMLInference } from '../src/api/hooks';

const mockPrediction = {
  mlPrediction: {
    symbol: 'AAPL',
    predicted_price: 155.20,
    confidence_interval: [153.50, 157.00],
    drift: 0.02,
    model_name: 'XGBoost-V4-Optimized',
    last_updated: '2026-03-19T00:00:00Z',
  },
};

test('MLPredictions renders prediction data correctly', async () => {
  vi.mocked(useMLInference).mockReturnValue({
    data: mockPrediction,
    loading: false,
    error: null,
  } as any);

  render(
    <ThemeProvider theme={theme}>
      <MLPredictions symbol="AAPL" />
    </ThemeProvider>
  );

  expect(await screen.findByText(/\$155\.20/)).toBeInTheDocument();
  expect(screen.getByText(/XGBoost-V4-Optimized/i)).toBeInTheDocument();
  expect(screen.getByText(/\+2\.00%/)).toBeInTheDocument();
});