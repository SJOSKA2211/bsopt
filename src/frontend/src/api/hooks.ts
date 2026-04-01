import { useQuery, useSubscription, gql } from '@apollo/client';
import { useQuery as useReactQuery, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';
import { useEffect, useRef } from 'react';
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
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    //  This is a placeholder for a real auth token
    const token = "dummy-jwt-for-ws"; 
    
    const connect = () => {
      const wsUrl = `ws://${window.location.host}/api/v1/market/ws/market-data?token=${token}`;
      ws.current = new WebSocket(wsUrl);

      ws.current.onopen = () => {
        console.log('Market data WebSocket connected');
        // Subscribe to initial symbols
        symbols.forEach(symbol => {
          ws.current?.send(JSON.stringify({ action: 'subscribe', symbol }));
        });
      };

      ws.current.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.type === 'ticker') {
          queryClient.setQueryData<Ticker[]>(['market', 'tickers'], (oldData) => {
            const newData = oldData ? [...oldData] : [];
            const index = newData.findIndex(t => t.symbol === message.data.symbol);
            if (index !== -1) {
              newData[index] = { ...newData[index], ...message.data };
            } else {
              newData.push(message.data);
            }
            return newData;
          });
        }
      };

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      ws.current.onclose = () => {
        console.log('Market data WebSocket disconnected. Reconnecting...');
        setTimeout(connect, 5000); // Reconnect after 5s
      };
    };

    connect();

    return () => {
      ws.current?.close();
    };
  }, [symbols, queryClient]);

  return useReactQuery<Ticker[]>({
    queryKey: ['market', 'tickers'],
    queryFn: () => [], // Data is managed by WebSocket
    staleTime: Infinity,
  });
}
