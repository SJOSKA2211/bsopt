import '@testing-library/jest-dom';
import { vi, beforeAll, afterAll, afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';
import { Server } from 'mock-socket';

// Clean up after each test
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

// Mock ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// Mock MatchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

// Mock canvas and 2D context
HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue({
  beginPath: vi.fn(),
  moveTo: vi.fn(),
  lineTo: vi.fn(),
  stroke: vi.fn(),
  fillRect: vi.fn(),
  clearRect: vi.fn(),
  arc: vi.fn(),
  fill: vi.fn(),
  measureText: vi.fn().mockReturnValue({ width: 0 }),
  createLinearGradient: vi.fn().mockReturnValue({ addColorStop: vi.fn() }),
});

// Mock Lightweight Charts
vi.mock('lightweight-charts', () => ({
  createChart: vi.fn().mockReturnValue({
    addAreaSeries: vi.fn().mockReturnValue({
      setData: vi.fn(),
      update: vi.fn(),
      applyOptions: vi.fn(),
    }),
    addLineSeries: vi.fn().mockReturnValue({
      setData: vi.fn(),
      update: vi.fn(),
      applyOptions: vi.fn(),
    }),
    addHistogramSeries: vi.fn().mockReturnValue({
      setData: vi.fn(),
      update: vi.fn(),
      applyOptions: vi.fn(),
    }),
    subscribeCrosshairMove: vi.fn(),
    unsubscribeCrosshairMove: vi.fn(),
    applyOptions: vi.fn(),
    resize: vi.fn(),
    remove: vi.fn(),
    timeScale: vi.fn().mockReturnValue({
      applyOptions: vi.fn(),
      fitContent: vi.fn(),
    }),
  }),
  ColorType: { Solid: 'solid' },
  CrosshairMode: { Normal: 0 },
}));

// Mock Three.js/Fiber
vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: any) => <div data-testid="r3f-canvas">{children}</div>,
  useFrame: vi.fn(),
  useThree: vi.fn().mockReturnValue({
    viewport: { width: 100, height: 100, factor: 1 },
    size: { width: 800, height: 600 },
  }),
  extend: vi.fn(),
}));

vi.mock('@react-three/drei', () => ({
  OrbitControls: () => <div data-testid="orbit-controls" />,
  PerspectiveCamera: () => <div data-testid="perspective-camera" />,
  Text: ({ children }: any) => <span>{children}</span>,
  Float: ({ children }: any) => <div>{children}</div>,
  Html: ({ children }: any) => <div>{children}</div>,
}));

// Global WebSocket and other mocks
export {};
