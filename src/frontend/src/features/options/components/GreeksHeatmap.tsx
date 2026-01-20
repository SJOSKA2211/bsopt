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
}

export const GreeksHeatmap: React.FC<GreeksHeatmapProps> = React.memo(({ symbol, greek }) => {
  const theme = useTheme();

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

  const chartOptions = useMemo(() => {
    if (!optionsData || optionsData.length === 0) return null;

    const strikes = Array.from(new Set(optionsData.map((d: OptionData) => d.strike))).sort((a: number, b: number) => a - b);
    const expiries = Array.from(new Set(optionsData.map((d: OptionData) => d.expiry))).sort();

    const data = optionsData.map((d: OptionData) => {
      const strikeIdx = strikes.indexOf(d.strike);
      const expiryIdx = expiries.indexOf(d.expiry);
      
      let value = 0;
      if (greek === 'delta') value = d.call_delta;
      else if (greek === 'gamma') value = d.call_gamma;
      else if (greek === 'iv') value = d.call_iv;

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
