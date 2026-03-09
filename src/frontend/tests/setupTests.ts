// src/frontend/setupTests.ts
import '@testing-library/jest-dom';
import { vi } from 'vitest';

// Mock lightweight-charts globally for tests
vi.mock('lightweight-charts', () => ({
  createChart: vi.fn(() => ({
    addLineSeries: vi.fn(() => ({
      setData: vi.fn(),
      update: vi.fn(),
    })),
    addCandlestickSeries: vi.fn(() => ({
      setData: vi.fn(),
      update: vi.fn(),
    })),
    addSeries: vi.fn(() => ({
      setData: vi.fn(),
      update: vi.fn(),
    })),
    applyOptions: vi.fn(),
    timeScale: vi.fn(() => ({
      fitContent: vi.fn(),
    })),
    remove: vi.fn(),
    resize: vi.fn(),
  })),
  ColorType: { Solid: 'solid' },
  CrosshairMode: { Normal: 0 },
  CandlestickSeries: "CandlestickSeries",
}));

// Mock ResizeObserver which is used by lightweight-charts and echarts but not present in jsdom
global.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
};

// Mock Worker for WASM-based pricing
global.Worker = class {
  onmessage: ((ev: MessageEvent) => any) | null = null;
  onmessageerror: ((ev: MessageEvent) => any) | null = null;
  onerror: ((ev: ErrorEvent) => any) | null = null;
  postMessage() {}
  terminate() {}
  addEventListener() {}
  removeEventListener() {}
  dispatchEvent() { return true; }
} as any;

// Mock Canvas getContext
HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
  fillRect: vi.fn(),
  clearRect: vi.fn(),
  getImageData: vi.fn(),
  putImageData: vi.fn(),
  createImageData: vi.fn(),
  setTransform: vi.fn(),
  drawImage: vi.fn(),
  save: vi.fn(),
  fillText: vi.fn(),
  restore: vi.fn(),
  beginPath: vi.fn(),
  moveTo: vi.fn(),
  lineTo: vi.fn(),
  closePath: vi.fn(),
  stroke: vi.fn(),
  translate: vi.fn(),
  scale: vi.fn(),
  rotate: vi.fn(),
  arc: vi.fn(),
  fill: vi.fn(),
  measureText: vi.fn(() => ({ width: 0 })),
  transform: vi.fn(),
  rect: vi.fn(),
  clip: vi.fn(),
})) as any;

// Mock matchMedia which is used by some MUI components
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(), // Deprecated
    removeListener: vi.fn(), // Deprecated
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
});

// Mock Apollo Client globally for components that do not have their own provider mock
vi.mock('@apollo/client/react', async (importOriginal) => {
  const actual: any = await importOriginal();
  return {
    ...actual,
    useQuery: vi.fn(() => ({ data: undefined, loading: true, error: undefined, refetch: vi.fn() })),
    useSubscription: vi.fn(() => ({ data: undefined, loading: true, error: undefined })),
    useMutation: vi.fn(() => [vi.fn(), { data: undefined, loading: false, error: undefined }]),
  };
});