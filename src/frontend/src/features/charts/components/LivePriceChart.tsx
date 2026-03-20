import React, { useEffect, useRef } from 'react';
import { Box, useTheme, alpha } from '@mui/material';
import { createChart, ColorType, CrosshairMode, CandlestickSeries } from 'lightweight-charts';
import type { IChartApi, ISeriesApi, Time } from 'lightweight-charts';
import { useQuery } from '@apollo/client/react';
import { gql } from '@apollo/client';
import { usePricingStore } from '../../../store/usePricingStore';

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

  // Access the live price tick directly from store
  const priceData = usePricingStore((state) => state.prices[symbol]);


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
      console.log(`[Zenith] No historical data for ${symbol}. Waiting for live ticks.`);
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
    if (priceData && seriesRef.current) {
      // OPTIMIZED: Update candlestick with live tick
      // If we don't have open/high/low/close, we treat the last price as the close
      seriesRef.current.update({
        time: (Math.floor(priceData.timestamp / 1000) as Time),
        open: priceData.price,
        high: priceData.price,
        low: priceData.price,
        close: priceData.price,
      });
    }
  }, [priceData]);


  return (
    <Box
      data-testid="live-price-chart-container"
      ref={chartContainerRef}
      sx={{ width: '100%', height: '100%', minHeight: 400 }}
    />
  );
};