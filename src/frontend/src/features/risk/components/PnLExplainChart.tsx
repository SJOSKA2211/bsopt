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
      axisPointer: { type: 'shadow' },
      backgroundColor: alpha('#0f172a', 0.9),
      borderColor: alpha(theme.palette.primary.main, 0.3),
      textStyle: { color: '#fff', fontFamily: 'JetBrains Mono', fontSize: 12 }
    },
    grid: { left: '3%', right: '4%', bottom: '3%', top: '10%', containLabel: true },
    xAxis: {
      type: 'category',
      data: ['Delta', 'Gamma', 'Vega', 'Theta', 'Residual', 'Total'],
      axisLine: { lineStyle: { color: alpha(theme.palette.divider, 0.2) } },
      axisLabel: { color: theme.palette.text.secondary, fontWeight: 700, fontSize: 10 }
    },
    yAxis: {
      type: 'value',
      splitLine: { lineStyle: { color: alpha(theme.palette.divider, 0.05) } },
      axisLabel: { color: theme.palette.text.secondary, fontFamily: 'JetBrains Mono' }
    },
    series: [
      {
        name: 'Placeholder',
        type: 'bar',
        stack: 'Total',
        itemStyle: { borderColor: 'transparent', color: 'transparent' },
        emphasis: { itemStyle: { borderColor: 'transparent', color: 'transparent' } },
        data: [0, data.delta, data.delta + data.gamma, data.delta + data.gamma + data.vega, data.delta + data.gamma + data.vega + data.theta, 0]
      },
      {
        name: 'Contribution',
        type: 'bar',
        stack: 'Total',
        label: { show: true, position: 'top', formatter: (p: any) => `$${p.value >= 0 ? '+' : ''}${p.value.toFixed(1)}` },
        itemStyle: {
          borderRadius: 4,
          color: (p: any) => {
            if (p.name === 'Total') return theme.palette.primary.main;
            return p.value >= 0 ? theme.palette.success.main : theme.palette.error.main;
          }
        },
        data: [
          data.delta, 
          data.gamma, 
          data.vega, 
          data.theta, 
          data.total - (data.delta + data.gamma + data.vega + data.theta),
          data.total
        ]
      }
    ]
  };

  return (
    <Box sx={{ 
      p: 3, 
      height: '100%', 
      borderRadius: 6, 
      background: `linear-gradient(135deg, ${alpha('#0f172a', 0.4)}, ${alpha('#0f172a', 0.1)})`,
      border: `1px solid ${alpha(theme.palette.divider, 0.05)}`
    }}>
      <Typography variant="caption" sx={{ fontWeight: 900, color: 'text.secondary', letterSpacing: '0.15em', textTransform: 'uppercase', mb: 2, display: 'block' }}>
        Attribution Waterfall (USD)
      </Typography>
      <ReactECharts option={option} style={{ height: '320px' }} />
    </Box>
  );
};
