import { expect, test, vi } from 'vitest';
import { registerServiceWorker } from '../src/pwa';

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

test.skip('Service worker registration is attempted by the application', async () => {
  registerServiceWorker();
  
  // We expect the real application code to call navigator.serviceWorker.register
  expect(mockRegister).toHaveBeenCalledWith('/service-worker.js');
});
