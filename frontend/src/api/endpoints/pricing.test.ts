/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest';
import { setupServer } from 'msw/node';
import { http, HttpResponse } from 'msw';
import { priceOption, calculateGreeks, PriceRequest, GreeksRequest } from '@/api/endpoints/pricing';

const handlers = [
  http.post('/api/v1/pricing/price', async ({ request }) => {
    const body = await request.json() as PriceRequest;
    return HttpResponse.json({
      success: true,
      data: {
        price: 5.67,
        spot: body.spot,
        strike: body.strike,
        time_to_expiry: body.time_to_expiry,
        rate: body.rate,
        volatility: body.volatility,
        option_type: body.option_type,
        model: body.model || 'black_scholes',
        timestamp: new Date().toISOString(),
        computation_time_ms: 1.2
      }
    });
  }),
  
  http.post('/api/v1/pricing/greeks', async ({ request }) => {
    const body = await request.json() as GreeksRequest;
    return HttpResponse.json({
      success: true,
      data: {
        delta: 0.52,
        gamma: 0.04,
        theta: -0.01,
        vega: 0.18,
        rho: 0.02,
        option_price: 5.67,
        spot: body.spot,
        strike: body.strike,
        time_to_expiry: body.time_to_expiry,
        volatility: body.volatility,
        option_type: body.option_type,
        timestamp: new Date().toISOString()
      }
    });
  }),
];

const server = setupServer(...handlers);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('Pricing API Endpoints', () => {
  it('should successfully price an option', async () => {
    const requestData: PriceRequest = {
      spot: 100,
      strike: 105,
      time_to_expiry: 0.5,
      rate: 0.05,
      volatility: 0.2,
      option_type: 'call'
    };
    
    const result = await priceOption(requestData);
    
    expect(result.price).toBe(5.67);
    expect(result.spot).toBe(100);
    expect(result.option_type).toBe('call');
  });

  it('should successfully calculate greeks', async () => {
    const requestData: GreeksRequest = {
      spot: 100,
      strike: 105,
      time_to_expiry: 0.5,
      rate: 0.05,
      volatility: 0.2,
      option_type: 'call'
    };
    
    const result = await calculateGreeks(requestData);
    
    expect(result.delta).toBe(0.52);
    expect(result.vega).toBe(0.18);
    expect(result.option_price).toBe(5.67);
  });
});
