import React, { useEffect, useRef, useState, useMemo } from 'react';
import { Box, useTheme, alpha, IconButton, Tooltip, Stack, Typography, CircularProgress, Chip } from '@mui/material';
import { 
  createChart, 
  ColorType, 
  CrosshairMode, 
  CandlestickSeries, 
  LineSeries,
  HistogramSeries
} from 'lightweight-charts';
import type { IChartApi, ISeriesApi, Time, CandlestickData, MouseEventParams } from 'lightweight-charts';
import { usePricingStore } from '../../../store/usePricingStore';
import type { PricingState } from '../../../store/usePricingStore';
import { useHistoricalData } from '../../../api/hooks';
import { Timeline, Flare as LiveIcon } from '@mui/icons-material';

interface LivePriceChartProps {
  symbol: string;
}

export const LivePriceChart: React.FC<LivePriceChartProps> = ({ symbol }: LivePriceChartProps) => {
  const theme = useTheme();
  const financial = (theme.palette as any).financial;
  const qfd = financial?.qfd;

  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const smaRef = useRef<ISeriesApi<"Line"> | null>(null);
  const volumeRef = useRef<ISeriesApi<"Histogram"> | null>(null);

  const [showSMA, setShowSMA] = useState(true);
  const [legendData, setLegendData] = useState<CandlestickData<Time> | null>(null);

  // Fetch historical data via unified hook
  const { data: historicalData, loading } = useHistoricalData(symbol);

  // Access the live price tick directly from store
  const priceData = usePricingStore((state: PricingState) => state.prices[symbol]);

  const smaData = useMemo(() => {
    if (!historicalData?.historicalData) return [];
    const data = historicalData.historicalData;
    const period = 20;
    const result = [];
    for (let i = period - 1; i < data.length; i++) {
      let sum = 0;
      for (let j = 0; j < period; j++) {
        sum += data[i - j].close;
      }
      result.push({
        time: data[i].time as Time,
        value: sum / period,
      });
    }
    return result;
  }, [historicalData]);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: chartContainerRef.current.clientHeight,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: theme.palette.text.secondary,
        fontFamily: 'Outfit, sans-serif',
      },
      grid: {
        vertLines: { visible: false },
        horzLines: { color: alpha(theme.palette.divider, 0.05) },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          labelBackgroundColor: qfd?.emerald ?? '#10b981',
          width: 1,
          style: 3, // Large dashed
          color: alpha(theme.palette.text.primary, 0.2),
        },
        horzLine: {
          labelBackgroundColor: qfd?.emerald ?? '#10b981',
          width: 1,
          style: 3,
          color: alpha(theme.palette.text.primary, 0.2),
        },
      },
      rightPriceScale: {
        borderColor: 'transparent',
        autoScale: true,
        alignLabels: true,
      },
      timeScale: {
        borderColor: 'transparent',
        timeVisible: true,
        secondsVisible: false,
        barSpacing: 10,
      },
      handleScale: {
        axisPressedMouseMove: true,
      },
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: qfd?.emerald ?? '#10b981',
      downColor: theme.palette.error.main,
      borderVisible: false,
      wickUpColor: qfd?.emerald ?? '#10b981',
      wickDownColor: theme.palette.error.main,
    });

    const volumeSeries = chart.addSeries(HistogramSeries, {
      color: alpha(qfd?.sky ?? '#38bdf8', 0.15),
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: '', // set as an overlay
    });
    
    volumeSeries.priceScale().applyOptions({
      scaleMargins: {
        top: 0.85,
        bottom: 0,
      },
    });

    const smaLine = chart.addSeries(LineSeries, {
      color: qfd?.sky ?? '#38bdf8',
      lineWidth: 2,
      visible: showSMA,
      lineType: 2, // Curved
    });

    // Subscribing to crosshair move for legend
    chart.subscribeCrosshairMove((param: MouseEventParams) => {
      if (param.time) {
        const data = param.seriesData.get(candleSeries) as CandlestickData<Time>;
        if (data) setLegendData(data);
      } else {
        setLegendData(null);
      }
    });

    // Set historical data when loaded
    if (historicalData?.historicalData) {
      const hData = historicalData.historicalData.map((d: { time: string | number; open: number; high: number; low: number; close: number }) => ({
        time: d.time as Time,
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
      }));
      candleSeries.setData(hData);

      const vData = historicalData.historicalData.map((d: { time: string | number; close: number; open: number; volume: number }) => ({
        time: d.time as Time,
        value: d.volume,
        color: d.close >= d.open ? alpha(qfd?.emerald ?? '#10b981', 0.2) : alpha(theme.palette.error.main, 0.2),
      }));
      volumeSeries.setData(vData);

      if (smaData.length > 0) {
        smaLine.setData(smaData);
      }
    }

    chartRef.current = chart;
    seriesRef.current = candleSeries;
    smaRef.current = smaLine;
    volumeRef.current = volumeSeries;

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
  }, [theme, historicalData, smaData]);

  // Update SMA visibility
  useEffect(() => {
    if (smaRef.current) {
      smaRef.current.applyOptions({ visible: showSMA });
    }
  }, [showSMA]);

  // Update chart when new data arrives
  useEffect(() => {
    if (priceData && seriesRef.current) {
      const timestamp = (Math.floor(priceData.timestamp / 1000) as Time);
      seriesRef.current.update({
        time: timestamp,
        open: priceData.price,
        high: Math.max(priceData.price, (priceData as any).high || 0),
        low: Math.min(priceData.price, (priceData as any).low || 1000000),
        close: priceData.price,
      });

      // High-precision volume data from Production feed
      if (volumeRef.current) {
        volumeRef.current.update({
          time: timestamp,
          value: priceData.volume || 0,
          color: alpha(qfd?.sky ?? '#38bdf8', 0.15),
        });
      }
    }
  }, [priceData, qfd?.sky]);

  return (
    <Box
      sx={{ 
        width: '100%', 
        height: '100%', 
        minHeight: 450, 
        position: 'relative',
        bgcolor: alpha(theme.palette.background.paper, 0.1),
        borderRadius: 6,
        border: `1px solid ${alpha('#fff', 0.05)}`,
        overflow: 'hidden',
        backdropFilter: 'blur(20px)',
      }}
    >
      {/* Header & Legend Overlay */}
      <Stack 
        direction="row" 
        justifyContent="space-between"
        alignItems="flex-start"
        sx={{ 
          position: 'absolute', 
          top: 16, 
          left: 16, 
          right: 16,
          zIndex: 10,
          pointerEvents: 'none'
        }}
      >
        <Stack spacing={0.5} sx={{ pointerEvents: 'auto' }}>
          <Stack direction="row" spacing={1} alignItems="center">
            <Box sx={{ width: 8, height: 8, bgcolor: qfd?.emerald ?? '#10b981', borderRadius: '50%', boxShadow: `0 0 10px ${qfd?.emerald ?? '#10b981'}` }} />
            <Typography variant="h6" sx={{ fontWeight: 900, fontFamily: 'Outfit', letterSpacing: '-0.02em' }}>
              {symbol}
            </Typography>
            <Chip 
              icon={<LiveIcon sx={{ fontSize: '10px !important' }} />}
              label="RDMA-LIVE" 
              size="small" 
              sx={{ 
                height: 18, 
                fontSize: '0.6rem', 
                bgcolor: alpha(qfd?.emerald ?? '#10b981', 0.1),
                color: qfd?.emerald,
                border: `1px solid ${alpha(qfd?.emerald ?? '#10b981', 0.2)}`,
                fontWeight: 800,
                '& .MuiChip-icon': { color: qfd?.emerald }
              }} 
            />
          </Stack>
          
          <Stack direction="row" spacing={2} sx={{ mt: 1 }}>
            {legendData ? (
              <>
                <Typography variant="caption" sx={{ fontFamily: 'JetBrains Mono', color: 'text.secondary', fontSize: '0.65rem' }}>
                  O <span style={{ color: theme.palette.text.primary, fontWeight: 700 }}>{legendData.open.toFixed(2)}</span>
                </Typography>
                <Typography variant="caption" sx={{ fontFamily: 'JetBrains Mono', color: 'text.secondary', fontSize: '0.65rem' }}>
                  H <span style={{ color: theme.palette.text.primary, fontWeight: 700 }}>{legendData.high.toFixed(2)}</span>
                </Typography>
                <Typography variant="caption" sx={{ fontFamily: 'JetBrains Mono', color: 'text.secondary', fontSize: '0.65rem' }}>
                  L <span style={{ color: theme.palette.text.primary, fontWeight: 700 }}>{legendData.low.toFixed(2)}</span>
                </Typography>
                <Typography variant="caption" sx={{ fontFamily: 'JetBrains Mono', color: 'text.secondary', fontSize: '0.65rem' }}>
                  C <span style={{ color: legendData.close >= legendData.open ? qfd?.emerald : theme.palette.error.main, fontWeight: 900 }}>{legendData.close.toFixed(2)}</span>
                </Typography>
              </>
            ) : priceData && (
              <Typography variant="h5" sx={{ fontFamily: 'JetBrains Mono', fontWeight: 900, color: qfd?.emerald }}>
                ${priceData.price.toFixed(2)}
              </Typography>
            )}
          </Stack>
        </Stack>

        <Stack direction="row" spacing={1} sx={{ pointerEvents: 'auto' }}>
          <Tooltip title="Toggle Production Trendline">
            <IconButton 
              aria-label="Toggle Production Trendline"
              size="small" 
              onClick={() => setShowSMA(!showSMA)}
              sx={{ 
                bgcolor: alpha(qfd?.sky ?? '#38bdf8', showSMA ? 0.15 : 0.05),
                color: showSMA ? qfd?.sky : 'text.disabled',
                border: `1px solid ${alpha(qfd?.sky ?? '#38bdf8', 0.2)}`,
                '&:hover': { bgcolor: alpha(qfd?.sky ?? '#38bdf8', 0.2) }
              }}
            >
              <Timeline sx={{ fontSize: 18 }} />
            </IconButton>
          </Tooltip>
        </Stack>
      </Stack>

      {/* Chart Container */}
      <Box
        data-testid="live-price-chart-container"
        ref={chartContainerRef}
        sx={{ 
          width: '100%', 
          height: '100%',
          '& .tv-lightweight-charts': {
            cursor: 'crosshair !important'
          }
        }}
      />
      
      {loading && (
        <Box sx={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', textAlign: 'center' }}>
          <CircularProgress size={32} thickness={5} sx={{ color: qfd?.emerald }} aria-label="Loading live price chart data" />
          <Typography variant="caption" sx={{ display: 'block', mt: 1, fontWeight: 800, color: 'text.secondary' }}>SYNCHRONIZING...</Typography>
        </Box>
      )}
    </Box>
  );
};