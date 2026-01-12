// src/frontend/src/mocks/handlers.ts
import { http, HttpResponse } from 'msw';

export const handlers = [
  http.get('/api/v1/options/chain', ({ request }) => {
    const url = new URL(request.url);
    const symbol = url.searchParams.get('symbol');
    const expiry = url.searchParams.get('expiry');

    // Mock data for options chain
    const mockOptionsChain = [
      {
        id: '1', strike: 100, expiry: '2026-03-01', underlying_price: 100.50,
        call_bid: 1.50, call_ask: 1.60, call_last: 1.55, call_volume: 100, call_oi: 500, call_iv: 0.20, call_delta: 0.55, call_gamma: 0.05,
        put_bid: 0.50, put_ask: 0.60, put_last: 0.55, put_volume: 80, put_oi: 400, put_iv: 0.22, put_delta: -0.45, put_gamma: 0.04,
      },
      {
        id: '2', strike: 105, expiry: '2026-03-01', underlying_price: 100.50,
        call_bid: 0.80, call_ask: 0.90, call_last: 0.85, call_volume: 120, call_oi: 600, call_iv: 0.18, call_delta: 0.30, call_gamma: 0.06,
        put_bid: 1.20, put_ask: 1.30, put_last: 1.25, put_volume: 110, put_oi: 550, put_iv: 0.25, put_delta: -0.70, put_gamma: 0.03,
      },
    ];

    // Filter by symbol and expiry (basic example)
    let filteredData = mockOptionsChain;
    if (symbol) {
      // In a real scenario, this would filter by symbol relevant options
    }
    if (expiry && expiry !== 'all') {
      // In a real scenario, this would filter by expiry date
    }

    return HttpResponse.json(filteredData, { status: 200 });
  }),
];
