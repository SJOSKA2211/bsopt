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
        const price = hestonEngine!.price_call(
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
        // The WASM binding for batch_calculate expects a specific format or array
        // Assuming the WASM binding handles the array of objects or we map it here
        // Since we can't easily pass complex objects to raw WASM without Serde, 
        // we might iterate here or rely on the binding's ability to take a JsValue (array).
        const result = engine.batch_calculate(payload);
        self.postMessage({ type: 'BATCH_CALCULATE_RESULT', payload: result, id });
        break;
      }

      case 'BATCH_PRICE_AMERICAN': {
        const { payload, id } = e.data as any;
        const result = engine.batch_price_american(new Float64Array(payload), 200, 200);
        // 🚀 SINGULARITY: Zero-copy transfer using Transferable Objects
        self.postMessage({ type: 'BATCH_CALCULATE_RESULT', payload: result, id }, [result.buffer]);
        break;
      }

      case 'BATCH_PRICE_MONTE_CARLO': {
        const { payload, id } = e.data as any;
        const result = engine.batch_price_monte_carlo(new Float64Array(payload), 100000);
        // 🚀 SINGULARITY: Zero-copy transfer using Transferable Objects
        self.postMessage({ type: 'BATCH_CALCULATE_RESULT', payload: result, id }, [result.buffer]);
        break;
      }

      case 'BATCH_PRICE_HESTON': {
        const { payload, id } = e.data as any;
        const result = engine.batch_price_heston(new Float64Array(payload));
        // 🚀 SINGULARITY: Zero-copy transfer using Transferable Objects
        self.postMessage({ type: 'BATCH_CALCULATE_RESULT', payload: result, id }, [result.buffer]);
        break;
      }
    }
  } catch (error) {
    self.postMessage({ type: 'ERROR', error: String(error), id: (e.data as any).id });
  }
};
