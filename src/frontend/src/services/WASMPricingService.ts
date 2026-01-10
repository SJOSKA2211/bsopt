import init, { BlackScholesWASM } from 'bsopt-wasm';

export interface OptionParams {
  spot: f64;
  strike: f64;
  time: f64;
  vol: f64;
  rate: f64;
  div: f64;
}

// Map f64 to number for TS
type f64 = number;

export interface Greeks {
  delta: number;
  gamma: number;
  vega: number;
  theta: number;
  rho: number;
}

export class WASMPricingService {
  private bs: BlackScholesWASM | null = null;
  private initialized: boolean = false;
  private initializing: Promise<void> | null = null;

  async initialize(): Promise<void> {
    if (this.initialized) return;
    if (this.initializing) return this.initializing;

    this.initializing = (async () => {
      try {
        await init();
        this.bs = new BlackScholesWASM();
        this.initialized = true;
        console.log('✅ WASM pricing engine initialized');
      } catch (error) {
        console.error('❌ Failed to initialize WASM pricing engine:', error);
        throw error;
      } finally {
        this.initializing = null;
      }
    })();

    return this.initializing;
  }

  isInitialized(): boolean {
    return this.initialized;
  }

  async priceCallOption(params: OptionParams): Promise<number> {
    if (!this.initialized) {
      await this.initialize();
    }
    return this.bs!.price_call(
      params.spot,
      params.strike,
      params.time,
      params.vol,
      params.rate,
      params.div
    );
  }

  async calculateGreeks(params: OptionParams): Promise<Greeks> {
    if (!this.initialized) {
      await this.initialize();
    }
    return this.bs!.calculate_greeks(
      params.spot,
      params.strike,
      params.time,
      params.vol,
      params.rate,
      params.div
    );
  }

  async priceOptionsBatch(options: OptionParams[]): Promise<Float64Array> {
    if (!this.initialized) {
      await this.initialize();
    }
    const stride = 7;
    const input = new Float64Array(options.length * stride);
    for (let i = 0; i < options.length; i++) {
      const opt = options[i];
      const offset = i * stride;
      input[offset] = opt.spot;
      input[offset + 1] = opt.strike;
      input[offset + 2] = opt.time;
      input[offset + 3] = opt.vol;
      input[offset + 4] = opt.rate;
      input[offset + 5] = opt.div;
      input[offset + 6] = 1.0; // is_call = true for now in this batch helper
    }
    return this.bs!.batch_calculate_compact(input);
  }
}

export const wasmPricing = new WASMPricingService();
