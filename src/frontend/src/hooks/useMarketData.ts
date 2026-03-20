import { useSubscription } from '@apollo/client/react';
import { gql } from '@apollo/client';
import { useMemo } from 'react';

const MARKET_DATA_SUBSCRIPTION = gql`
  subscription OnMarketData($symbols: [String!]!) {
    marketDataStream(symbols: $symbols) {
      symbol
      lastPrice: last_price
      volume
    }
  }
`;

export interface MarketTick {
    symbol: string;
    lastPrice: number;
    volume: number;
}

/**
 * Hook to subscribe to real-time market data ticks for one or more symbols.
 */
console.log('[useMarketData] Script loaded');

export const useMarketData = (symbols: string | string[]) => {
    const symbolList = useMemo(() =>
        typeof symbols === 'string' ? [symbols] : symbols,
        [symbols]
    );

    const { data, loading, error } = useSubscription(MARKET_DATA_SUBSCRIPTION, {
        variables: { symbols: symbolList },
        skip: symbolList.length === 0,
    });

    return {
        tick: (data as { marketDataStream?: MarketTick })?.marketDataStream,
        isLoading: loading,
        isError: !!error,
        error,
    };
};
