import { useEffect, useState, useRef, useCallback } from 'react';

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

export const useWasmPricing = () => {
  const [isLoaded, setIsLoaded] = useState(false);
  const workerRef = useRef<Worker | null>(null);
  const pendingRequests = useRef<Map<string, (resolve: any, reject: any) => void>>(new Map());

  useEffect(() => {
    // Initialize Web Worker
    const worker = new Worker(new URL('../workers/pricing.worker.ts', import.meta.url), {
      type: 'module'
    });
    
    worker.onmessage = (e) => {
      const { type, payload, id, error } = e.data;
      
      if (type === 'INIT_SUCCESS') {
        setIsLoaded(true);
        console.log('WASM Worker initialized successfully');
        return;
      }

      if (id && pendingRequests.current.has(id)) {
        const resolver = pendingRequests.current.get(id);
        pendingRequests.current.delete(id);
        
        if (error) {
          resolver?.reject(error);
        } else {
          resolver?.resolve(payload);
        }
      }
    };

    worker.postMessage({ type: 'INIT' });
    workerRef.current = worker;

    return () => {
      worker.terminate();
    };
  }, []);

  const _sendWorkerMessage = useCallback((type: string, payload: any): Promise<any> => {
    return new Promise((resolve, reject) => {
      if (!workerRef.current || !isLoaded) {
        // Fallback or early return if worker not ready
        resolve(null); 
        return;
      }
      
      const id = Math.random().toString(36).substring(7);
      pendingRequests.current.set(id, { resolve, reject });
      workerRef.current.postMessage({ type, payload, id });
    });
  }, [isLoaded]);

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