import { render, screen, waitFor } from '@testing-library/react';
import { expect, test, vi, beforeAll, afterEach, afterAll } from 'vitest';
import { VolatilitySurface3D } from '../src/features/options/components/VolatilitySurface3D';
import { ThemeProvider } from '@mui/material/styles';
import { theme } from '../src/theme/index';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import React from 'react';

// Mock React Three Fiber and Three.js
vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: any) => <div data-testid="three-canvas-mock">{children}</div>,
  useFrame: vi.fn(),
  useThree: vi.fn(() => ({ viewport: { width: 100, height: 100 } })),
}));

vi.mock('@react-three/drei', () => ({
  OrbitControls: () => <div data-testid="orbit-controls-mock" />,
  PerspectiveCamera: () => <div data-testid="camera-mock" />,
  Text: () => <div data-testid="text-mock" />,
}));

// Mock Server Setup
const handlers = [
  http.get('/api/v1/options/chain', () => {
    return HttpResponse.json([
      { strike: 100, expiry: '2026-03-01', call_iv: 0.2 },
    ]);
  }),
];

const server = setupServer(...handlers);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
};

test('VolatilitySurface3D renders canvas container', async () => {
  render(
    <ThemeProvider theme={theme}>
      <VolatilitySurface3D symbol="AAPL" />
    </ThemeProvider>,
    { wrapper: createWrapper() }
  );

  await waitFor(() => {
    expect(screen.getByTestId('volatility-surface-container')).toBeInTheDocument();
    expect(screen.getByTestId('three-canvas-mock')).toBeInTheDocument();
  }, { timeout: 2000 });
});
