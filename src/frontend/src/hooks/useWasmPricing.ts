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

// Initialize worker (lazy)
const getWorker = () => {
  if (sharedWorker) return sharedWorker;

  // Initialize Web Worker
  sharedWorker = new Worker(new URL('../workers/pricing.worker.ts', import.meta.url), {
    type: 'module'
  });

  sharedWorker.onmessage = (e) => {
    const { type, payload, id, error } = e.data;

    if (type === 'INIT_SUCCESS') {
      isWorkerReady = true;
      console.log('WASM Worker initialized successfully');
      listeners.forEach(listener => listener());
      return;
    }

    if (id && pendingRequests.has(id)) {
      const resolver = pendingRequests.get(id);
      pendingRequests.delete(id);

      if (error) {
        resolver?.reject(error);
      } else {
        resolver?.resolve(payload);
      }
    }
  };

  sharedWorker.postMessage({ type: 'INIT' });
  return sharedWorker;
};

// External store subscription
const subscribe = (callback: () => void) => {
  listeners.add(callback);
  return () => {
    listeners.delete(callback);
  };
};

const getSnapshot = () => isWorkerReady;

const useWasmPricing = () => {
  const isLoaded = useSyncExternalStore(subscribe, getSnapshot);

  useEffect(() => {
    // Ensure worker is initialized
    getWorker();
  }, []);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const _sendWorkerMessage = useCallback((type: string, payload: any): Promise<any> => {
    return new Promise((resolve, reject) => {
      const worker = getWorker();
      if (!worker || !isWorkerReady) {
        // Fallback or early return if worker not ready
        resolve(null);
        return;
      }

      const id = Math.random().toString(36).substring(7);
      pendingRequests.set(id, { resolve, reject });
      worker.postMessage({ type, payload, id });
    });
  }, []);

  const priceOption = useCallback(async (params: OptionParams): Promise<OptionResult | null> => {
    return _sendWorkerMessage('PRICE_OPTION', params);
  }, [_sendWorkerMessage]);

  const calculateIV = useCallback(async (price: number, params: Omit<OptionParams, 'vol'>): Promise<number | null> => {
    return _sendWorkerMessage('CALCULATE_IV', { price, ...params });
  }, [_sendWorkerMessage]);

  const batchCalculate = useCallback(async (params: OptionParams[]): Promise<OptionResult[]> => {
     return _sendWorkerMessage('BATCH_CALCULATE', params) as Promise<OptionResult[]>;
  }, [_sendWorkerMessage]);

  const priceAmerican = useCallback(async (params: OptionParams, m?: number, n?: number): Promise<{ price: number } | null> => {
    return _sendWorkerMessage('PRICE_AMERICAN', { ...params, m, n });
  }, [_sendWorkerMessage]);

  const priceMonteCarlo = useCallback(async (params: OptionParams, num_paths?: number): Promise<{ price: number } | null> => {
    return _sendWorkerMessage('PRICE_MONTE_CARLO', { ...params, num_paths });
  }, [_sendWorkerMessage]);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const priceHeston = useCallback(async (params: any): Promise<{ price: number } | null> => {
    return _sendWorkerMessage('PRICE_HESTON', params);
  }, [_sendWorkerMessage]);

  return {
    isLoaded,
    priceOption,
    calculateIV,
    batchCalculate,
    priceAmerican,
    priceMonteCarlo,
    priceHeston
  };
};

export { useWasmPricing };
if (typeof window !== 'undefined') {
  (window as any).useWasmPricing = useWasmPricing;
}

