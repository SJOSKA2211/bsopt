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
}

export const wasmPricing = new WASMPricingService();
