import React, { useEffect, useRef } from 'react';
import { Box, useTheme, alpha } from '@mui/material';
import { createChart, ColorType, CrosshairMode, CandlestickSeries } from 'lightweight-charts';
import type { IChartApi, ISeriesApi, Time } from 'lightweight-charts';
import { useWebSocket } from '../../../hooks/useWebSocket';

interface LivePriceChartProps {
  symbol: string;
}

export const LivePriceChart: React.FC<LivePriceChartProps> = ({ symbol }) => {
  const theme = useTheme();
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  // Hook into real-time data
  const { data: wsData } = useWebSocket<{
    symbol: string;
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
  }>({
    url: 'ws://localhost:1234/marketdata',
    symbols: [symbol],
    enabled: true
  }); // Placeholder URL

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: chartContainerRef.current.clientHeight,
      layout: {
        background: { type: ColorType.Solid, color: theme.palette.background.paper },
        textColor: theme.palette.text.secondary,
      },
      grid: {
        vertLines: { color: alpha(theme.palette.divider, 0.1) },
        horzLines: { color: alpha(theme.palette.divider, 0.1) },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      rightPriceScale: {
        borderColor: alpha(theme.palette.divider, 0.2),
      },
      timeScale: {
        borderColor: alpha(theme.palette.divider, 0.2),
        timeVisible: true,
        secondsVisible: false,
      },
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: theme.palette.success.main,
      downColor: theme.palette.error.main,
      borderVisible: false,
      wickUpColor: theme.palette.success.main,
      wickDownColor: theme.palette.error.main,
    });

    // Mock historical data for now
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
  }, [theme]);

  // Update chart when new data arrives
  useEffect(() => {
    if (wsData && wsData.symbol === symbol && seriesRef.current) {
      seriesRef.current.update({
        time: (wsData.time as Time),
        open: wsData.open,
        high: wsData.high,
        low: wsData.low,
        close: wsData.close,
      });
    }
  }, [wsData, symbol]);

  return (
    <Box
      data-testid="live-price-chart-container"
      ref={chartContainerRef}
      sx={{ width: '100%', height: '100%', minHeight: 400 }}
    />
  );
};