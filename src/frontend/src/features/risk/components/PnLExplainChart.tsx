import React from 'react';
import ReactECharts from 'echarts-for-react';
import { Box, Typography, useTheme } from '@mui/material';

interface PnLExplainChartProps {
  data: {
    delta: number;
    gamma: number;
    vega: number;
    theta: number;
    total: number;
  };
}

export const PnLExplainChart: React.FC<PnLExplainChartProps> = ({ data }) => {
  const theme = useTheme();

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' }
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'value',
      splitLine: { lineStyle: { type: 'dashed', color: theme.palette.divider } }
    },
    yAxis: {
      type: 'category',
      data: ['Delta', 'Gamma', 'Vega', 'Theta', 'Total'],
      axisLine: { lineStyle: { color: theme.palette.text.secondary } }
    },
    series: [
      {
        name: 'P&L Contribution ($)',
        type: 'bar',
        label: { show: true, position: 'right', formatter: '${c}' },
        data: [
          { value: data.delta, itemStyle: { color: theme.palette.info.main } },
          { value: data.gamma, itemStyle: { color: theme.palette.warning.main } },
          { value: data.vega, itemStyle: { color: theme.palette.secondary.main } },
          { value: data.theta, itemStyle: { color: theme.palette.error.main } },
          { value: data.total, itemStyle: { color: theme.palette.primary.main, fontWeight: 'bold' } }
        ]
      }
    ]
  };

  return (
    <Box sx={{ p: 2, height: '100%' }}>
      <Typography variant="h6" gutterBottom>Institutional P&L Explain (Real-time)</Typography>
      <ReactECharts option={option} style={{ height: '300px' }} />
    </Box>
  );
};
