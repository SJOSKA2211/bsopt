import { gql } from '@apollo/client';
import { useQuery } from '@apollo/client/react';
import { useQuery as useReactQuery, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';
import { useEffect } from 'react';
import type { MarketData, MLPrediction, OptionConnection, Ticker, PortfolioSummary } from './types';

// Production GraphQL Fragments
const OPTION_FIELDS = gql`
  fragment OptionFields on Option {
    id
    symbol
    expiry
    strike
    type
    bid
    ask
    last_price
    volume
    open_interest
    implied_volatility
    greeks {
      delta
      gamma
      vega
      theta
      rho
    }
  }
`;

// Queries
export const GET_MARKET_DATA = gql`
  query GetMarketData($symbol: String!) {
    marketData(symbol: $symbol) {
      symbol
      last_price
      bid
      ask
      volume
      timestamp
    }
  }
`;

export const GET_OPTIONS = gql`
  ${OPTION_FIELDS}
  query GetOptions($symbol: String, $min_strike: Float, $max_strike: Float, $expiry: Date) {
    options(symbol: $symbol, min_strike: $min_strike, max_strike: $max_strike, expiry: $expiry) {
      edges {
        cursor
        node {
          ...OptionFields
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
`;

export const GET_ML_PREDICTION = gql`
  query GetMLPrediction($symbol: String!) {
    mlPrediction(symbol: $symbol) {
      id
      symbol
      predicted_price
      confidence_interval
      model_name
      timestamp
    }
  }
`;

// Subscriptions (Real-time Greeks)
export const GREEKS_SUBSCRIPTION = gql`
  subscription OnGreeksUpdate($symbols: [String!]!) {
    greeksUpdate(symbols: $symbols) {
      symbol
      delta
      gamma
      vega
      theta
      rho
    }
  }
`;

export const GET_HISTORICAL_DATA = gql`
  query GetHistoricalData($symbol: String!) {
    historicalData(symbol: $symbol) {
      time
      open
      high
      low
      close
      volume
    }
  }
`;

// Fused Hooks
export function useHistoricalData(symbol: string) {
  return useQuery<{ historicalData: any[] }>(GET_HISTORICAL_DATA, {
    variables: { symbol },
  });
}

export function useOptionsChain(symbol: string) {
  return useQuery<{ options: OptionConnection }>(GET_OPTIONS, {
    variables: { symbol },
    notifyOnNetworkStatusChange: true,
  });
}

export function useProductionMarketData(symbol: string) {
  return useQuery<{ marketData: MarketData }>(GET_MARKET_DATA, {
    variables: { symbol },
    pollInterval: 5000, // 5s poll as fallback to WS
  });
}

export function useMLInference(symbol: string) {
  return useQuery<{ mlPrediction: MLPrediction }>(GET_ML_PREDICTION, {
    variables: { symbol },
  });
}

// REST Integration (React Query)
const api = axios.create({
  baseURL: '/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

export function useLogin() {
  const queryClient = useQueryClient();
  return {
    mutateAsync: async (payload: any) => {
      const { data } = await api.post('/auth/login', payload);
      if (data.data?.access_token) {
        localStorage.setItem('access_token', data.data.access_token);
        localStorage.setItem('refresh_token', data.data.refresh_token);
      }
      return data;
    }
  };
}

export function useRegister() {
  return {
    mutateAsync: async (payload: any) => {
      const { data } = await api.post('/auth/register', payload);
      return data;
    }
  };
}

export function usePortfolioSummary() {
  return useReactQuery<PortfolioSummary>({
    queryKey: ['portfolio', 'summary'],
    queryFn: async () => {
      const { data } = await api.get('/portfolio/summary');
      return data;
    },
  });
}

export function useComparisonData() {
  return useReactQuery({
    queryKey: ['ml', 'comparison'],
    queryFn: async () => {
      const { data } = await api.get('/ml/comparison');
      return data;
    },
    refetchInterval: 5000,
  });
}

export function useLiveTickers(symbols: string[]) {
  const queryClient = useQueryClient();

  useEffect(() => {
    const symbolQuery = symbols.join(',');
    const sseUrl = `/api/v1/market/sse/market-data?symbols=${symbolQuery}`;
    
    let eventSource: EventSource | null = null;
    
    const connect = () => {
      eventSource = new EventSource(sseUrl);

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          queryClient.setQueryData<Ticker[]>(['market', 'tickers'], (oldData) => {
            const newData = oldData ? [...oldData] : [];
            const items = Array.isArray(data) ? data : [data];
            
            items.forEach((item: Ticker) => {
               const index = newData.findIndex(t => t.symbol === item.symbol);
               if (index !== -1) {
                 newData[index] = { ...newData[index], ...item };
               } else {
                 newData.push(item);
               }
            });
            return newData;
          });
        } catch (e) {
          console.error("SSE parse error:", e);
        }
      };

      eventSource.onerror = (error) => {
        console.error('SSE error:', error);
        eventSource?.close();
        setTimeout(connect, 5000);
      };
    };

    connect();

    return () => {
      eventSource?.close();
    };
  }, [symbols.join(','), queryClient]);

  return useReactQuery<Ticker[]>({
    queryKey: ['market', 'tickers'],
    queryFn: () => symbols.map(sym => ({ symbol: sym, price: '0.00', percentChange: '0.00%', up: true } as Ticker)), // Initial fallback
    staleTime: Infinity,
  });
}
export function useSignals(limit: number = 20) {
  return useReactQuery({
    queryKey: ['system', 'signals', limit],
    queryFn: async () => {
      const { data } = await api.get(`/system/signals?limit=${limit}`);
      return data;
    },
    refetchInterval: 5000,
  });
}
