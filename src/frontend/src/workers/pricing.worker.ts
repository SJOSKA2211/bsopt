/// <reference lib="webworker" />

import init, { BlackScholesWASM, MonteCarloWASM, CrankNicolsonWASM, HestonWASM } from '../wasm/bsopt_wasm';

// Define message types
type PricingMessage = 
  | { type: 'INIT' }
  | { type: 'PRICE_OPTION'; payload: any; id: string }
  | { type: 'PRICE_AMERICAN'; payload: any; id: string }
  | { type: 'PRICE_MONTE_CARLO'; payload: any; id: string }
  | { type: 'PRICE_HESTON'; payload: any; id: string }
  | { type: 'CALCULATE_IV'; payload: any; id: string }
  | { type: 'BATCH_CALCULATE'; payload: any[]; id: string }
  | { type: 'BATCH_PRICE_AMERICAN'; payload: number[]; id: string }
  | { type: 'BATCH_PRICE_MONTE_CARLO'; payload: number[]; id: string }
  | { type: 'BATCH_PRICE_HESTON'; payload: number[]; id: string };

let engine: BlackScholesWASM | null = null;
let mcEngine: MonteCarloWASM | null = null;
let cnEngine: CrankNicolsonWASM | null = null;
let hestonEngine: HestonWASM | null = null;

const initializeWasm = async () => {
  try {
    await init();
    engine = new BlackScholesWASM();
    mcEngine = new MonteCarloWASM();
    cnEngine = new CrankNicolsonWASM();
    hestonEngine = new HestonWASM();
    self.postMessage({ type: 'INIT_SUCCESS' });
  } catch (error) {
    self.postMessage({ type: 'ERROR', error: String(error) });
  }
};

self.onmessage = async (e: MessageEvent<PricingMessage>) => {
  const { type } = e.data;

  if (type === 'INIT') {
    await initializeWasm();
    return;
  }

  if (!engine || !mcEngine || !cnEngine) {
    self.postMessage({ type: 'ERROR', error: 'WASM engine not initialized', id: (e.data as any).id });
    return;
  }

  try {
    switch (type) {
      case 'PRICE_OPTION': {
        const { payload, id } = e.data as any;
        const { spot, strike, time, vol, rate, div, is_call } = payload;
        const price = is_call 
          ? engine.price_call(spot, strike, time, vol, rate, div)
          : engine.price_put(spot, strike, time, vol, rate, div);
        const greeks = engine.calculate_greeks(spot, strike, time, vol, rate, div);
        
        self.postMessage({ type: 'PRICE_OPTION_RESULT', payload: { price, greeks }, id });
        break;
      }

      case 'PRICE_AMERICAN': {
        const { payload, id } = e.data as any;
        const { spot, strike, time, vol, rate, div, is_call, m, n } = payload;
        const price = cnEngine.price_american(
          spot, strike, time, vol, rate, div, is_call, 
          m || 200, n || 200
        );
        self.postMessage({ type: 'PRICE_OPTION_RESULT', payload: { price }, id });
        break;
      }

      case 'PRICE_MONTE_CARLO': {
        const { payload, id } = e.data as any;
        const { spot, strike, time, vol, rate, div, is_call, num_paths } = payload;
        // @ts-ignore
        const price = mcEngine.price_european(
          spot, strike, time, vol, rate, div, is_call, 
          num_paths || 100000
        );
        self.postMessage({ type: 'PRICE_OPTION_RESULT', payload: { price }, id });
        break;
      }

      case 'PRICE_HESTON': {
        const { payload, id } = e.data as any;
        const { spot, strike, time, r, v0, kappa, theta, sigma, rho } = payload;

        // HestonWASM.price_call expects: (spot, strike, time, r, v0, kappa, theta, sigma, rho)
        // If TS error persists, force cast
        const price = (hestonEngine as any).price_call(
          spot, strike, time, r, v0, kappa, theta, sigma, rho
        );
        self.postMessage({ type: 'PRICE_OPTION_RESULT', payload: { price }, id });
        break;
      }
      
      case 'CALCULATE_IV': {
        const { payload, id } = e.data as any;
        const { price, spot, strike, time, rate, div, is_call } = payload;
        const result = engine.solve_iv(price, spot, strike, time, rate, div, is_call);
        self.postMessage({ type: 'CALCULATE_IV_RESULT', payload: result, id });
        break;
      }

      case 'BATCH_CALCULATE': {
        const { payload, id } = e.data as any;
        // Updated method name to match WASM interface
        // @ts-ignore
        const result = engine.batch_calculate_soa(payload);
        self.postMessage({ type: 'BATCH_CALCULATE_RESULT', payload: result, id });
        break;
      }

      case 'BATCH_PRICE_AMERICAN': {
        const { payload, id } = e.data as any;
        const data = new Float64Array(payload);
        const result = engine.batch_price_american(data, 200, 200);
        // Transfer buffer ownership for zero-copy efficiency
        self.postMessage({ type: 'BATCH_CALCULATE_RESULT', payload: result, id }, [result.buffer]);
        break;
      }

      case 'BATCH_PRICE_MONTE_CARLO': {
        const { payload, id } = e.data as any;
        const data = new Float64Array(payload);
        const result = engine.batch_price_monte_carlo(data, 100000);
        // Transfer buffer ownership for zero-copy efficiency
        self.postMessage({ type: 'BATCH_CALCULATE_RESULT', payload: result, id }, [result.buffer]);
        break;
      }

      case 'BATCH_PRICE_HESTON': {
        const { payload, id } = e.data as any;
        const data = new Float64Array(payload);
        // Fixed argument count for batch_price_heston
        // @ts-ignore
        const result = engine.batch_price_heston(data);
        // Transfer buffer ownership for zero-copy efficiency
        self.postMessage({ type: 'BATCH_CALCULATE_RESULT', payload: result, id }, [result.buffer]);
        break;
      }
    }
  } catch (error) {
    self.postMessage({ type: 'ERROR', error: String(error), id: (e.data as any).id });
  }
};
