import { wasmPricing, OptionParams } from './WASMPricingService';

export async function runBenchmarks(count: number = 1000) {
  await wasmPricing.initialize();
  
  const params: OptionParams = {
    spot: 100,
    strike: 100,
    time: 1,
    vol: 0.2,
    rate: 0.05,
    div: 0,
  };

  const optionParams: OptionParams[] = Array.from({ length: count }, () => ({ ...params }));

  // 1. Individual WASM Calls
  const startIndividual = performance.now();
  for (let i = 0; i < count; i++) {
    await wasmPricing.priceCallOption(optionParams[i]);
  }
  const endIndividual = performance.now();
  const individualTime = endIndividual - startIndividual;

  // 2. Batch WASM Call (Compact)
  const startBatch = performance.now();
  await wasmPricing.priceOptionsBatch(optionParams);
  const endBatch = performance.now();
  const batchTime = endBatch - startBatch;

  // 3. Mock API Call (Simulation)
  // Simulate 50ms network latency + backend processing for a SINGLE call
  // For 'count' calls, it would be much worse, but we compare one batch API call
  const mockApiTime = 50 + (Math.random() * 20);

  return {
    individualTime,
    batchTime,
    mockApiTime,
    count,
  };
}