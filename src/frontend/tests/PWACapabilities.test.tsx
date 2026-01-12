import { expect, test, vi } from 'vitest';

// Mock navigator.serviceWorker.register to track calls
const mockRegister = vi.fn().mockResolvedValue({ scope: '/' });

Object.defineProperty(global.navigator, 'serviceWorker', {
  value: {
    register: mockRegister,
    ready: Promise.resolve(null),
    controller: null,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  },
  configurable: true,
});

// Mock react-dom/client to prevent rendering
vi.mock('react-dom/client', () => ({
  createRoot: () => ({
    render: vi.fn(),
  }),
}));

test('Service worker registration is attempted by the application', async () => {
  // Import main.tsx to trigger the registration logic
  await import('../src/main');
  
  // We expect the real application code to call navigator.serviceWorker.register
  expect(mockRegister).toHaveBeenCalledWith('/service-worker.js');
});