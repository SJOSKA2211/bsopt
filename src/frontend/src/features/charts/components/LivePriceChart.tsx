import React, { useEffect, useRef, useState, useMemo } from 'react';
import { Box, useTheme, alpha, IconButton, Tooltip, Stack, Typography, CircularProgress } from '@mui/material';
import { 
  createChart, 
  ColorType, 
  CrosshairMode, 
  CandlestickSeries, 
  LineSeries,
  HistogramSeries
} from 'lightweight-charts';
import type { IChartApi, ISeriesApi, Time, CandlestickData } from 'lightweight-charts';
import { usePricingStore } from '../../../store/usePricingStore';
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
  const priceData = usePricingStore((state: any) => state.prices[symbol]);

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
        fontFamily: 'Inter, sans-serif',
      },
      grid: {
        vertLines: { color: alpha(theme.palette.divider, 0.03) },
        horzLines: { color: alpha(theme.palette.divider, 0.03) },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          labelBackgroundColor: theme.palette.primary.main,
        },
        horzLine: {
          labelBackgroundColor: theme.palette.primary.main,
        },
      },
      rightPriceScale: {
        borderColor: alpha(theme.palette.divider, 0.1),
        autoScale: true,
      },
      timeScale: {
        borderColor: alpha(theme.palette.divider, 0.1),
        timeVisible: true,
        secondsVisible: false,
      },
      handleScale: {
        axisPressedMouseMove: true,
      },
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: theme.palette.success.main,
      downColor: theme.palette.error.main,
      borderVisible: false,
      wickUpColor: theme.palette.success.main,
      wickDownColor: theme.palette.error.main,
    });

    const volumeSeries = chart.addSeries(HistogramSeries, {
      color: alpha(theme.palette.primary.main, 0.2),
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: '', // set as an overlay
    });
    
    volumeSeries.priceScale().applyOptions({
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });

    const smaLine = chart.addSeries(LineSeries, {
      color: theme.palette.primary.main,
      lineWidth: 2,
      visible: showSMA,
    });

    // Set historical data when loaded
    if (historicalData?.historicalData) {
      const hData = historicalData.historicalData.map((d: any) => ({
        time: d.time as Time,
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
      }));
      candleSeries.setData(hData);

      const vData = historicalData.historicalData.map((d: any) => ({
        time: d.time as Time,
        value: d.volume,
        color: d.close >= d.open ? alpha(theme.palette.success.main, 0.3) : alpha(theme.palette.error.main, 0.3),
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
        high: priceData.price,
        low: priceData.price,
        close: priceData.price,
      });

      // Simple mock volume for live ticks
      if (volumeRef.current) {
        volumeRef.current.update({
          time: timestamp,
          value: Math.random() * 100 + 50,
          color: alpha(theme.palette.primary.main, 0.3),
        });
      }
    }
  }, [priceData, theme]);

  return (
    <Box
      sx={{ 
        width: '100%', 
        height: '100%', 
        minHeight: 450, 
        position: 'relative',
        bgcolor: alpha(theme.palette.background.paper, 0.2),
        borderRadius: 4,
        border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
        overflow: 'hidden',
        backdropFilter: 'blur(10px)',
      }}
    >
      <Stack 
        direction="row" 
        spacing={2} 
        alignItems="center" 
        sx={{ 
          position: 'absolute', 
          top: 12, 
          left: 12, 
          zIndex: 10,
          bgcolor: alpha(theme.palette.background.default, 0.7),
          px: 1.5,
          py: 0.75,
          borderRadius: 3,
          backdropFilter: 'blur(12px)',
          border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
          boxShadow: `0 4px 20px ${alpha('#000', 0.4)}`
        }}
      >
        <Typography variant="caption" sx={{ fontWeight: 900, letterSpacing: '0.1em', color: 'primary.main', display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box sx={{ width: 6, height: 6, bgcolor: 'primary.main', borderRadius: '50%', animation: 'pulse 1.5s infinite' }} />
          LIVE: {symbol}
        </Typography>
        
        {/* Greeks Overlay Sub-Component */}
        <Stack direction="row" spacing={1.5} sx={{ borderLeft: `1px solid ${alpha(theme.palette.divider, 0.2)}`, pl: 1.5 }}>
          {['Δ', 'Γ', 'Θ', 'V'].map((g, idx) => (
             <Typography key={g} variant="caption" sx={{ fontFamily: 'JetBrains Mono', fontWeight: 800, color: 'text.secondary' }}>
               <span style={{ color: alpha(theme.palette.text.secondary, 0.5) }}>{g}:</span>
               <span style={{ color: theme.palette.text.primary, marginLeft: 2 }}>
                 {(Math.random() * (idx === 0 ? 0.5 : 0.05)).toFixed(3)}
               </span>
             </Typography>
          ))}
        </Stack>

        <Stack direction="row" spacing={0.5}>
          <Tooltip title="Toggle SMA (20)">
            <IconButton 
              size="small" 
              onClick={() => setShowSMA(!showSMA)}
              color={showSMA ? 'primary' : 'default'}
              sx={{ width: 28, height: 28, bgcolor: alpha(theme.palette.primary.main, showSMA ? 0.1 : 0) }}
            >
              <Timeline sx={{ fontSize: 18 }} />
            </IconButton>
          </Tooltip>
        </Stack>
      </Stack>

      <Box
        data-testid="live-price-chart-container"
        ref={chartContainerRef}
        sx={{ width: '100%', height: '100%' }}
      />
      
      {loading && (
        <Box sx={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}>
          <CircularProgress size={30} thickness={4} aria-label="Loading price chart..." />
        </Box>
      )}
    </Box>
  );
};