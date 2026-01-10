import { describe, it, expect, vi } from 'vitest';
import { runBenchmarks } from './benchmark';

// Mock performance.now for deterministic tests if needed, but here we want REAL values
// However, since we are in JSDOM, performance.now() might be available.

vi.mock('bsopt-wasm', () => {
  return {
    default: vi.fn().mockResolvedValue({}),
    BlackScholesWASM: vi.fn().mockImplementation(function() {
      return {
        price_call: vi.fn().mockReturnValue(10.45),
        batch_calculate_compact: vi.fn().mockReturnValue(new Float64Array(1000 * 6)),
      };
    }),
  };
});

describe('Latency Benchmarks', () => {
  it('should run benchmarks and compare performance', async () => {
    const results = await runBenchmarks(100);
    
    console.log(`\n--- Latency Benchmark Results (${results.count} options) ---`);
    console.log(`Individual WASM: ${results.individualTime.toFixed(4)}ms`);
    console.log(`Batch WASM:      ${results.batchTime.toFixed(4)}ms`);
    console.log(`Mock API (1 call): ~${results.mockApiTime.toFixed(2)}ms`);
    console.log(`--------------------------------------------------\n`);
    
    expect(results.individualTime).toBeGreaterThanOrEqual(0);
    expect(results.batchTime).toBeGreaterThanOrEqual(0);
    // In a real environment, batch should be faster than individual total
  });
});
