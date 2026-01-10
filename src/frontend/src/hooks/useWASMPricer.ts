import { useState, useEffect, useCallback } from 'react';
import { wasmPricing, OptionParams, Greeks } from '../services/WASMPricingService';

export function useWASMPricer() {
  const [isInitialized, setIsInitialized] = useState(wasmPricing.isInitialized());
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let isMounted = true;
    
    async function init() {
      try {
        await wasmPricing.initialize();
        if (isMounted) {
          setIsInitialized(true);
        }
      } catch (err) {
        if (isMounted) {
          setError(err instanceof Error ? err : new Error(String(err)));
        }
      }
    }

    if (!isInitialized) {
      init();
    }

    return () => {
      isMounted = false;
    };
  }, [isInitialized]);

  const priceCall = useCallback(async (params: OptionParams) => {
    return wasmPricing.priceCallOption(params);
  }, []);

  const calculateGreeks = useCallback(async (params: OptionParams): Promise<Greeks> => {
    return wasmPricing.calculateGreeks(params);
  }, []);

  return {
    isInitialized,
    error,
    priceCall,
    calculateGreeks,
  };
}
