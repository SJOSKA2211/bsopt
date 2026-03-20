/// <reference lib="webworker" />

/* eslint-disable @typescript-eslint/no-unused-vars */
import init, { BlackScholesWASM, MonteCarloWASM, CrankNicolsonWASM, HestonWASM } from '../wasm/bsopt_wasm';
/* eslint-enable @typescript-eslint/no-unused-vars */

interface OptionPayload {
  spot: number;
  strike: number;
  time: number;
  vol: number;
  rate: number;
  div: number;
  is_call: boolean;
  m?: number;
  n?: number;
  num_paths?: number;
}

interface HestonPayload {
  spot: number;
  strike: number;
  time: number;
  r: number;
  v0: number;
  kappa: number;
  theta: number;
  sigma: number;
  rho: number;
}

interface IVPayload {
  price: number;
  spot: number;
  strike: number;
  time: number;
  rate: number;
  div: number;
  is_call: boolean;
}

type PricingMessage = 
  | { type: 'INIT'; id?: string }
  | { type: 'PRICE_OPTION'; payload: OptionPayload; id: string }
  | { type: 'PRICE_AMERICAN'; payload: OptionPayload; id: string }
  | { type: 'PRICE_MONTE_CARLO'; payload: OptionPayload; id: string }
  | { type: 'PRICE_HESTON'; payload: HestonPayload; id: string }
  | { type: 'CALCULATE_IV'; payload: IVPayload; id: string }
  | { type: 'BATCH_CALCULATE'; payload: OptionPayload[]; id: string }
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

  if (!engine || !mcEngine || !cnEngine || !hestonEngine) {
    const id = 'id' in e.data ? e.data.id : undefined;
    self.postMessage({ type: 'ERROR', error: 'WASM engine not initialized', id });
    return;
  }

  try {
    switch (type) {
      case 'PRICE_OPTION': {
        const { payload, id } = e.data;
        const { spot, strike, time, vol, rate, div, is_call } = payload;
        const price = is_call 
          ? engine.price_call(spot, strike, time, vol, rate, div)
          : engine.price_put(spot, strike, time, vol, rate, div);
        const greeks = engine.calculate_greeks(spot, strike, time, vol, rate, div);
        
        self.postMessage({ type: 'PRICE_OPTION_RESULT', payload: { price, greeks }, id });
        break;
      }

      case 'PRICE_AMERICAN': {
        const { payload, id } = e.data;
        const { spot, strike, time, vol, rate, div, is_call, m, n } = payload;
        const price = engine.price_american(
          spot, strike, time, vol, rate, div, is_call, 
          m || 200, n || 200
        );
        self.postMessage({ type: 'PRICE_OPTION_RESULT', payload: { price }, id });
        break;
      }

      case 'PRICE_MONTE_CARLO': {
        const { payload, id } = e.data;
        const { spot, strike, time, vol, rate, div, is_call, num_paths } = payload;
        const price = engine.price_monte_carlo(
          spot, strike, time, vol, rate, div, is_call, 
          num_paths || 100000
        );
        self.postMessage({ type: 'PRICE_OPTION_RESULT', payload: { price }, id });
        break;
      }

      case 'PRICE_HESTON': {
        const { payload, id } = e.data;
        const { spot, strike, time, r, v0, kappa, theta, sigma, rho } = payload;
        const price = hestonEngine!.price_call(
          spot, strike, time, r, v0, kappa, theta, sigma, rho
        );
        self.postMessage({ type: 'PRICE_OPTION_RESULT', payload: { price }, id });
        break;
      }
      
      case 'CALCULATE_IV': {
        const { payload, id } = e.data;
        const { price, spot, strike, time, rate, div, is_call } = payload;
        const result = engine.solve_iv(price, spot, strike, time, rate, div, is_call);
        self.postMessage({ type: 'CALCULATE_IV_RESULT', payload: result, id });
        break;
      }

      case 'BATCH_CALCULATE': {
        const { payload: options, id } = e.data;
        const n = options.length;

        // Transform AoS to SoA for SIMD-accelerated WASM
        const spots = new Float64Array(n);
        const strikes = new Float64Array(n);
        const times = new Float64Array(n);
        const vols = new Float64Array(n);
        const rates = new Float64Array(n);
        const divs = new Float64Array(n);
        const areCalls = new Float64Array(n);

        options.forEach((opt, i) => {
          spots[i] = opt.spot;
          strikes[i] = opt.strike;
          times[i] = opt.time;
          vols[i] = opt.vol;
          rates[i] = opt.rate;
          divs[i] = opt.div;
          areCalls[i] = opt.is_call ? 1.0 : 0.0;
        });

        const rawResults = engine.batch_calculate_soa_compact(
          spots, strikes, times, vols, rates, divs, areCalls
        );

        // Reconstruct AoS results (Stride of 6: price, delta, gamma, vega, theta, rho)
        const results = [];
        for (let i = 0; i < n; i++) {
          const offset = i * 6;
          results.push({
            price: rawResults[offset],
            greeks: {
              delta: rawResults[offset + 1],
              gamma: rawResults[offset + 2],
              vega: rawResults[offset + 3],
              theta: rawResults[offset + 4],
              rho: rawResults[offset + 5]
            }
          });
        }

        self.postMessage({ type: 'BATCH_CALCULATE_RESULT', payload: results, id });
        break;
      }

      case 'BATCH_PRICE_AMERICAN': {
        const { payload, id } = e.data;
        const data = new Float64Array(payload);
        const result = engine.batch_price_american(data, 200, 200);
        self.postMessage({ type: 'BATCH_CALCULATE_RESULT', payload: result, id }, [result.buffer]);
        break;
      }

      case 'BATCH_PRICE_MONTE_CARLO': {
        const { payload, id } = e.data;
        const data = new Float64Array(payload);
        const result = engine.batch_price_monte_carlo(data, 10000);
        self.postMessage({ type: 'BATCH_CALCULATE_RESULT', payload: result, id }, [result.buffer]);
        break;
      }

      case 'BATCH_PRICE_HESTON': {
        const { payload, id } = e.data;
        const data = new Float64Array(payload);
        const result = engine.batch_price_heston(data);
        self.postMessage({ type: 'BATCH_CALCULATE_RESULT', payload: result, id }, [result.buffer]);
        break;
      }
    }
  } catch (error) {
    const id = 'id' in e.data ? e.data.id : undefined;
    self.postMessage({ type: 'ERROR', error: String(error), id });
  }
};
