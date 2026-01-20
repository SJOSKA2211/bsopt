import React from 'react';
import ReactECharts from 'echarts-for-react';
import { Box, Typography, useTheme } from '@mui/material';

interface VolSkewPlotProps {
  symbol: string;
  data: {
    strike: number;
    market_iv: number;
    svi_iv: number;
  }[];
}

export const VolSkewPlot: React.FC<VolSkewPlotProps> = ({ symbol, data }) => {
  const theme = useTheme();

  const option = {
    backgroundColor: 'transparent',
    title: {
      text: `${symbol} Volatility Skew (SVI vs Market)`,
      textStyle: { color: theme.palette.text.primary, fontSize: 14 },
      left: 'center',
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross' }
    },
    legend: {
      data: ['Market IV', 'SVI Fit'],
      bottom: 0,
      textStyle: { color: theme.palette.text.secondary }
    },
    xAxis: {
      type: 'value',
      name: 'Strike',
      splitLine: { lineStyle: { color: theme.palette.divider } },
      axisLabel: { color: theme.palette.text.secondary }
    },
    yAxis: {
      type: 'value',
      name: 'Implied Vol',
      splitLine: { lineStyle: { color: theme.palette.divider } },
      axisLabel: { color: theme.palette.text.secondary }
    },
    series: [
      {
        name: 'Market IV',
        data: data.map(d => [d.strike, d.market_iv]),
        type: 'scatter',
        itemStyle: { color: theme.palette.secondary.main }
      },
      {
        name: 'SVI Fit',
        data: data.map(d => [d.strike, d.svi_iv]),
        type: 'line',
        smooth: true,
        itemStyle: { color: theme.palette.primary.main }
      }
    ]
  };

  return (
    <Box sx={{ width: '100%', height: 300, p: 2 }}>
      <ReactECharts option={option} style={{ height: '100%' }} />
    </Box>
  );
};
