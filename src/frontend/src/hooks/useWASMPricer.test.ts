import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useWASMPricer } from './useWASMPricer';
import { wasmPricing } from '../services/WASMPricingService';

vi.mock('../services/WASMPricingService', () => {
  const mockService = {
    initialize: vi.fn().mockResolvedValue(undefined),
    isInitialized: vi.fn().mockReturnValue(false),
    priceCallOption: vi.fn().mockResolvedValue(10.45),
    calculateGreeks: vi.fn().mockResolvedValue({
      delta: 0.6368,
      gamma: 0.0187,
      vega: 0.3752,
      theta: -0.0175,
      rho: 0.5323,
    }),
  };
  return {
    wasmPricing: mockService,
    WASMPricingService: vi.fn().mockImplementation(() => mockService),
  };
});

describe('useWASMPricer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should initialize the WASM engine on mount', async () => {
    await act(async () => {
      renderHook(() => useWASMPricer());
    });
    expect(wasmPricing.initialize).toHaveBeenCalled();
  });

  it('should return the correct initialization status', () => {
    vi.mocked(wasmPricing.isInitialized).mockReturnValue(true);
    const { result } = renderHook(() => useWASMPricer());
    expect(result.current.isInitialized).toBe(true);
  });

  it('should provide pricing functions', async () => {
    const { result } = renderHook(() => useWASMPricer());
    
    let price;
    await act(async () => {
      price = await result.current.priceCall({
        spot: 100,
        strike: 100,
        time: 1,
        vol: 0.2,
        rate: 0.05,
        div: 0,
      });
    });
    
    expect(price).toBe(10.45);
    expect(wasmPricing.priceCallOption).toHaveBeenCalled();
  });
});
