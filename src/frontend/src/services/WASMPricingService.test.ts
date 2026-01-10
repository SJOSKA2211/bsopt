import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock the WASM module
vi.mock('bsopt-wasm', () => {
  return {
    default: vi.fn().mockResolvedValue({}),
    BlackScholesWASM: vi.fn().mockImplementation(function() {
      return {
        price_call: vi.fn().mockReturnValue(10.45),
        calculate_greeks: vi.fn().mockReturnValue({
          delta: 0.6368,
          gamma: 0.0187,
          vega: 0.3752,
          theta: -0.0175,
          rho: 0.5323,
        }),
      };
    }),
  };
});

import { WASMPricingService } from './WASMPricingService';

describe('WASMPricingService', () => {
  let service: WASMPricingService;

  beforeEach(() => {
    vi.clearAllMocks();
    service = new WASMPricingService();
  });

  it('should start as not initialized', () => {
    expect(service.isInitialized()).toBe(false);
  });

  it('should initialize successfully', async () => {
    await service.initialize();
    expect(service.isInitialized()).toBe(true);
  });

  it('should call WASM price_call when pricing a call option', async () => {
    await service.initialize();
    const price = await service.priceCallOption({
      spot: 100,
      strike: 100,
      time: 1,
      vol: 0.2,
      rate: 0.05,
      div: 0,
    });
    expect(price).toBe(10.45);
  });

  it('should automatically initialize if priceCallOption is called before initialize', async () => {
    const price = await service.priceCallOption({
      spot: 100,
      strike: 100,
      time: 1,
      vol: 0.2,
      rate: 0.05,
      div: 0,
    });
    expect(price).toBe(10.45);
    expect(service.isInitialized()).toBe(true);
  });
});
