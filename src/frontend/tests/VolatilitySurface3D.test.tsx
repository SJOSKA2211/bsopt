import { render, screen } from '@testing-library/react';
import { expect, test, vi } from 'vitest';
import { VolatilitySurface3D } from '../src/features/options/components/VolatilitySurface3D';
import { ThemeProvider } from '@mui/material/styles';
import { theme } from '../src/theme/index';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';

// Mock React Three Fiber and Three.js
vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: { children: React.ReactNode }) => <div data-testid="three-canvas-mock">{children}</div>,
  useFrame: vi.fn(),
  useThree: vi.fn(() => ({ viewport: { width: 100, height: 100 } })),
}));

vi.mock('@react-three/drei', () => ({
  OrbitControls: () => <div data-testid="orbit-controls-mock" />,
  PerspectiveCamera: () => <div data-testid="camera-mock" />,
  Text: () => <div data-testid="text-mock" />,
}));

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

test('VolatilitySurface3D renders canvas container', () => {
  render(
    <ThemeProvider theme={theme}>
      <VolatilitySurface3D symbol="AAPL" />
    </ThemeProvider>,
    { wrapper: createWrapper() }
  );

  expect(screen.getByTestId('volatility-surface-container')).toBeInTheDocument();
  expect(screen.getByTestId('three-canvas-mock')).toBeInTheDocument();
});