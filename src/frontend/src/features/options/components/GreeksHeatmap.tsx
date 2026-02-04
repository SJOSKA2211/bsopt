import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react/lib/core';
import * as echarts from 'echarts/core';
import { HeatmapChart } from 'echarts/charts';
import {
  TooltipComponent,
  GridComponent,
  VisualMapComponent
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import { Box, CircularProgress, Typography, useTheme, alpha } from '@mui/material';
import { useQuery } from '@tanstack/react-query';
import { useWasmPricing } from '../../../hooks/useWasmPricing';

// Register the required components
echarts.use([
  HeatmapChart,
  TooltipComponent,
  GridComponent,
  VisualMapComponent,
  CanvasRenderer
]);

interface GreeksHeatmapProps {
  symbol: string;
  greek: 'delta' | 'gamma' | 'iv' | 'theta' | 'vega';
}

interface OptionData {
  strike: number;
  expiry: string;
  call_delta: number;
  call_gamma: number;
  call_iv: number;
  put_delta: number;
  put_gamma: number;
  put_iv: number;
  underlying_price: number;
}

export const GreeksHeatmap: React.FC<GreeksHeatmapProps> = React.memo(({ symbol, greek }) => {
  const theme = useTheme();
  const { batchCalculate, isLoaded: isWasmLoaded } = useWasmPricing();

  const { data: optionsData, isLoading, error } = useQuery<OptionData[]>({
    queryKey: ['options-chain', symbol, 'all'],
    queryFn: async () => {
      const response = await fetch(`/api/v1/options/chain?symbol=${symbol}&expiry=all`);
      if (!response.ok) {
        throw new Error('Failed to fetch options data');
      }
      return response.json();
    },
  });

  // Enrich data with WASM for Greeks if loaded
  const processedData = useMemo(() => {
    if (!optionsData || !isWasmLoaded) return optionsData || [];

    // Generate all params for batch calculation
    const params = optionsData.map(d => ({
      spot: d.underlying_price,
      strike: d.strike,
      time: 30 / 365, // Mock time
      vol: d.call_iv,
      rate: 0.05,
      div: 0.0,
      is_call: true
    }));

    const results = batchCalculate(params);

    return optionsData.map((d, i) => {
      const result = results[i];
      if (result) {
        return {
          ...d,
          call_delta: result.greeks.delta,
          call_gamma: result.greeks.gamma,
          call_vega: result.greeks.vega,
          call_theta: result.greeks.theta,
        };
      }
      return d;
    });
  }, [optionsData, isWasmLoaded, batchCalculate]);

  const chartOptions = useMemo(() => {
    if (!processedData || processedData.length === 0) return null;

    const strikes = Array.from(new Set(processedData.map((d: OptionData) => d.strike))).sort((a: number, b: number) => a - b);
    const expiries = Array.from(new Set(processedData.map((d: OptionData) => d.expiry))).sort();

    const data = processedData.map((d: OptionData) => {
      const strikeIdx = strikes.indexOf(d.strike);
      const expiryIdx = expiries.indexOf(d.expiry);
      
      let value = 0;
      if (greek === 'delta') value = d.call_delta;
      else if (greek === 'gamma') value = d.call_gamma;
      else if (greek === 'iv') value = d.call_iv;
      else if (greek === 'theta') value = (d as any).call_theta || 0;
      else if (greek === 'vega') value = (d as any).call_vega || 0;

      return [strikeIdx, expiryIdx, value];
    });

    return {
      tooltip: {
        position: 'top',
        formatter: (params: { data: [number, number, number] }) => {
          return `Strike: $${strikes[params.data[0]]}<br/>Expiry: ${expiries[params.data[1]]}<br/>${greek.toUpperCase()}: ${params.data[2].toFixed(4)}`;
        }
      },
      grid: {
        height: '80%',
        top: '10%',
        left: '10%',
        right: '5%'
      },
      xAxis: {
        type: 'category',
        data: strikes.map(s => `$${s}`),
        splitArea: { show: true },
        axisLabel: { color: theme.palette.text.secondary }
      },
      yAxis: {
        type: 'category',
        data: expiries,
        splitArea: { show: true },
        axisLabel: { color: theme.palette.text.secondary }
      },
      visualMap: {
        min: 0,
        max: 1,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '0%',
        inRange: {
          color: [
            alpha(theme.palette.primary.main, 0.1),
            theme.palette.primary.main,
            theme.palette.secondary.main
          ]
        },
        textStyle: { color: theme.palette.text.secondary }
      },
      series: [{
        name: `${greek.toUpperCase()} Heatmap`,
        type: 'heatmap',
        data: data,
        label: { show: false },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        }
      }],
      backgroundColor: 'transparent'
    };
  }, [optionsData, greek, theme]);

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', minHeight: 300 }}>
        <CircularProgress aria-label="Loading Greeks heatmap" />
      </Box>
    );
  }

  if (error || !optionsData) {
    return (
      <Box sx={{ p: 4, textAlign: 'center' }}>
        <Typography color="error">Error loading heatmap data</Typography>
      </Box>
    );
  }

  return (
    <Box data-testid="greeks-heatmap-container" sx={{ width: '100%', height: '100%', minHeight: 400 }}>
      {chartOptions && (
        <ReactECharts 
          echarts={echarts}
          option={chartOptions} 
          style={{ height: '100%', width: '100%' }}
          theme={theme.palette.mode === 'dark' ? 'dark' : undefined}
        />
      )}
    </Box>
  );
});
