import React, { useEffect, useRef } from 'react';
import { Box, useTheme, alpha } from '@mui/material';
import { createChart, ColorType, CrosshairMode, CandlestickSeries } from 'lightweight-charts';
import type { IChartApi, ISeriesApi, Time } from 'lightweight-charts';
import { useSubscription, useQuery } from '@apollo/client/react';
import { gql } from '@apollo/client';

const MARKET_SUBSCRIPTION = gql`
  subscription OnMarketUpdate($symbols: [String!]!) {
    market_data_stream(symbols: $symbols) {
      symbol
      lastPrice: last_price
      volume
    }
  }
`;

const GET_HISTORICAL_DATA = gql`
  query GetHistoricalData($symbol: String!) {
    historicalData(symbol: $symbol) {
      time
      open
      high
      low
      close
    }
  }
`;


interface LivePriceChartProps {
  symbol: string;
}

export const LivePriceChart: React.FC<LivePriceChartProps> = ({ symbol }) => {
  const theme = useTheme();
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  // Fetch historical data
  const { data: historicalData } = useQuery(GET_HISTORICAL_DATA, {
    variables: { symbol },
  });

  // Sub for live updates
  const { data: subData } = useSubscription(MARKET_SUBSCRIPTION, {
    variables: { symbols: [symbol] },
  });


  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: chartContainerRef.current.clientHeight,
      layout: {
        background: { type: ColorType.Solid, color: '#0f172a' }, // Solid Slate 900
        textColor: theme.palette.text.secondary,
      },
      grid: {
        vertLines: { color: alpha(theme.palette.divider, 0.05) },
        horzLines: { color: alpha(theme.palette.divider, 0.05) },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      rightPriceScale: {
        borderColor: alpha(theme.palette.divider, 0.1),
      },
      timeScale: {
        borderColor: alpha(theme.palette.divider, 0.1),
        timeVisible: true,
        secondsVisible: false,
      },
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#10b981',
      downColor: '#f43f5e',
      borderVisible: false,
      wickUpColor: '#10b981',
      wickDownColor: '#f43f5e',
    });

    // Set historical data when loaded
    if ((historicalData as any)?.historicalData) {
      candleSeries.setData((historicalData as any).historicalData.map((d: { time: Time, open: number, high: number, low: number, close: number }) => ({
        ...d,
        time: (d.time as Time)
      })));
    } else {
      // Mock historical data if query fails or is empty for demo
      const data = [];
      const now = new Date();
      for (let i = 0; i < 100; i++) {
        const time = new Date(now.getTime() - (100 - i) * 60000);
        data.push({
          time: (Math.floor(time.getTime() / 1000) as Time),
          open: 150 + Math.random() * 10,
          high: 165 + Math.random() * 10,
          low: 145 + Math.random() * 10,
          close: 155 + Math.random() * 10,
        });
      }
      candleSeries.setData(data);
    }

    chartRef.current = chart;
    seriesRef.current = candleSeries;


    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.resize(chartContainerRef.current.clientWidth, chartContainerRef.current.clientHeight);
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [theme, historicalData]);

  // Update chart when new data arrives
  useEffect(() => {
    if ((subData as any)?.market_data_stream && seriesRef.current) {
      const update = (subData as any).market_data_stream;
      seriesRef.current.update({
        time: (Math.floor(Date.now() / 1000) as Time),
        value: update.lastPrice,
      } as any); // Chart is currently candlestick, but subscription only provides last price
    }
  }, [subData]);


  return (
    <Box
      data-testid="live-price-chart-container"
      ref={chartContainerRef}
      sx={{ width: '100%', height: '100%', minHeight: 400 }}
    />
  );
};