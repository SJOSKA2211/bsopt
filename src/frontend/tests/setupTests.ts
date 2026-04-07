// src/frontend/setupTests.ts
import '@testing-library/jest-dom';
import React from 'react';
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
Object.defineProperty(globalThis, 'ResizeObserver', {
  value: class {
    observe() {}
    unobserve() {}
    disconnect() {}
  },
  writable: true
});

// Mock Worker for WASM-based pricing
Object.defineProperty(globalThis, 'Worker', {
  value: class {
    onmessage: ((ev: MessageEvent) => unknown) | null = null;
    onmessageerror: ((ev: MessageEvent) => unknown) | null = null;
    onerror: ((ev: ErrorEvent) => unknown) | null = null;
    postMessage() {}
    terminate() {}
    addEventListener() {}
    removeEventListener() {}
    dispatchEvent() { return true; }
  },
  writable: true
});

// Mock Canvas getContext
(HTMLCanvasElement.prototype.getContext as unknown) = vi.fn(() => ({
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
})) as unknown as RenderingContext;

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
  })),
});

// Mock Apollo Client globally for components that do not have their own provider mock
vi.mock('@apollo/client/react', async (importOriginal) => {
  const actual = await importOriginal() as Record<string, unknown>;
  return {
    ...actual,
    useQuery: vi.fn(() => ({ data: undefined, loading: true, error: undefined, refetch: vi.fn() })),
    useSubscription: vi.fn(() => ({ data: undefined, loading: true, error: undefined })),
    useMutation: vi.fn(() => [vi.fn(), { data: undefined, loading: false, error: undefined }]),
  };
});

vi.mock('@react-three/drei', () => ({
  Points: ({ children }: any) => React.createElement('div', { 'data-testid': 'drei-points' }, children),
  PointMaterial: () => null,
  Float: ({ children }: any) => React.createElement('div', { 'data-testid': 'drei-float' }, children),
}));

// Mock Three.js and R3F to prevent unrecognized tag errors in jsdom
vi.mock('three', () => ({
  WebGLRenderer: vi.fn(() => ({
    setSize: vi.fn(),
    render: vi.fn(),
  })),
  Scene: vi.fn(),
  PerspectiveCamera: vi.fn(),
  Mesh: vi.fn(),
  Group: vi.fn(),
  Points: vi.fn(),
  BoxGeometry: vi.fn(),
  MeshStandardMaterial: vi.fn(),
  DirectionalLight: vi.fn(),
  AmbientLight: vi.fn(),
  PointLight: vi.fn(() => ({ position: { set: vi.fn() } })),
  Vector3: vi.fn(() => ({ set: vi.fn(), lerp: vi.fn() })),
  Color: vi.fn(),
  AdditiveBlending: 2,
  ShaderMaterial: vi.fn(),
  MathUtils: {
    lerp: (a: number, b: number, t: number) => a + (b - a) * t,
  },
}));

vi.mock('@react-three/fiber', async (importOriginal) => {
  const actual = await importOriginal() as Record<string, unknown>;
  return {
    ...actual,
    Canvas: ({ children }: { children: React.ReactNode }) => React.createElement('div', { 'data-testid': 'canvas' }, children),
    useFrame: vi.fn(),
    useThree: vi.fn(() => ({
      viewport: { width: 1000, height: 1000, factor: 1 },
      size: { width: 1000, height: 1000 },
      mouse: { x: 0, y: 0 },
      camera: { position: { x: 0, y: 0, z: 8 } },
    })),
  };
});if (typeof globalThis.IntersectionObserver === 'undefined') {
  globalThis.IntersectionObserver = class IntersectionObserver {
    root: Element | Document | null = null;
    rootMargin: string = '';
    thresholds: ReadonlyArray<number> = [];
    disconnect() {}
    observe() {}
    takeRecords() { return []; }
    unobserve() {}
  };
}
