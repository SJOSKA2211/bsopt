import { useEffect, useCallback, useSyncExternalStore } from 'react';

// Interface matching the Rust structs
export interface OptionParams {
  spot: number;
  strike: number;
  time: number;
  vol: number;
  rate: number;
  div: number;
  is_call: boolean;
}

export interface Greeks {
  delta: number;
  gamma: number;
  vega: number;
  theta: number;
  rho: number;
}

export interface OptionResult {
  price: number;
  greeks: Greeks;
}

// Singleton state
let sharedWorker: Worker | null = null;
const pendingRequests = new Map<string, { resolve: (val: unknown) => void; reject: (err: unknown) => void }>();
let isWorkerReady = false;
const listeners = new Set<() => void>();

let requestIdCounter = 0;
const generateId = () => `req_${Date.now()}_${requestIdCounter++}`;

// Initialize worker (lazy)
const getWorker = () => {
  if (sharedWorker) return sharedWorker;

  try {
    sharedWorker = new Worker(new URL('../workers/pricing.worker.ts', import.meta.url), {
      type: 'module'
    });

    sharedWorker.onmessage = (e) => {
      const { type, payload, id, error } = e.data;

      if (type === 'INIT_SUCCESS') {
        isWorkerReady = true;
        console.log('[WASM] Worker initialized');
        listeners.forEach(listener => listener());
        return;
      }

      if (id && pendingRequests.has(id)) {
        const resolver = pendingRequests.get(id);
        pendingRequests.delete(id);
        if (error) resolver?.reject(error);
        else resolver?.resolve(payload);
      }
    };

    sharedWorker.postMessage({ type: 'INIT' });
  } catch (err) {
    console.error('[WASM] Failed to create worker:', err);
  }
  
  return sharedWorker;
};

const subscribe = (callback: () => void) => {
  listeners.add(callback);
  return () => listeners.delete(callback);
};

const getSnapshot = () => isWorkerReady;

export const useWasmPricing = () => {
  const isLoaded = useSyncExternalStore(subscribe, getSnapshot);

  useEffect(() => {
    getWorker();
  }, []);

  const _callWorker = useCallback((type: string, payload: any): Promise<any> => {
    return new Promise((resolve, reject) => {
      const worker = getWorker();
      if (!worker || !isWorkerReady) {
        resolve(null);
        return;
      }
      const id = generateId();
      pendingRequests.set(id, { resolve, reject });
      worker.postMessage({ type, payload, id });
    });
  }, []);

  return {
    isLoaded,
    priceOption: (params: OptionParams) => _callWorker('PRICE_OPTION', params),
    calculateIV: (price: number, params: Omit<OptionParams, 'vol'>) => 
        _callWorker('CALCULATE_IV', { price, ...params }),
    batchCalculate: (params: OptionParams[]) => _callWorker('BATCH_CALCULATE', params),
    priceAmerican: (params: OptionParams, m?: number, n?: number) => 
        _callWorker('PRICE_AMERICAN', { ...params, m, n }),
    priceMonteCarlo: (params: OptionParams, num_paths?: number) => 
        _callWorker('PRICE_MONTE_CARLO', { ...params, num_paths }),
    priceHeston: (params: any) => _callWorker('PRICE_HESTON', params),
    batchPriceAmerican: (params: number[], m?: number, n?: number) => 
        _callWorker('BATCH_PRICE_AMERICAN', { payload: params, m, n }),
    batchPriceMonteCarlo: (params: number[], num_paths?: number) => 
        _callWorker('BATCH_PRICE_MONTE_CARLO', { payload: params, num_paths }),
    batchPriceHeston: (params: number[]) => _callWorker('BATCH_PRICE_HESTON', { payload: params }),
  };
};
