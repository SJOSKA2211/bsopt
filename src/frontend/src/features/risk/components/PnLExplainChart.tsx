import React from 'react';
import ReactECharts from 'echarts-for-react';
import { Box, Typography, useTheme, alpha } from '@mui/material';

interface PnLExplainChartProps {
  data: {
    delta: number;
    gamma: number;
    vega: number;
    theta: number;
    total: number;
  };
}

export const PnLExplainChart: React.FC<PnLExplainChartProps> = ({ data }: PnLExplainChartProps) => {
  const theme = useTheme();
  const financial = (theme.palette as any).financial;
  const qfd = financial?.qfd;

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: alpha('#0f172a', 0.9),
      borderColor: alpha(qfd?.emerald ?? '#10b981', 0.3),
      borderWidth: 1,
      textStyle: { 
        color: '#fff', 
        fontFamily: 'JetBrains Mono, monospace', 
        fontSize: 12,
        fontWeight: 600
      },
      padding: [10, 15],
      formatter: (params: any[]) => {
        const p = params[1]; // Real data
        return `<div style="font-family: Outfit; font-weight: 800; margin-bottom: 4px; color: ${alpha('#fff', 0.5)}; text-transform: uppercase; font-size: 10px;">Attribution</div>
                <div style="display: flex; justify-content: space-between; gap: 20px; align-items: center;">
                  <span style="font-weight: 700;">${p.name}</span>
                  <span style="font-family: 'JetBrains Mono'; font-weight: 800; color: ${p.value >= 0 ? qfd?.emerald : theme.palette.error.main}">
                    $${p.value >= 0 ? '+' : ''}${p.value.toFixed(2)}
                  </span>
                </div>`;
      }
    },
    grid: { left: '3%', right: '4%', bottom: '5%', top: '15%', containLabel: true },
    xAxis: {
      type: 'category',
      data: ['Delta', 'Gamma', 'Vega', 'Theta', 'Residual', 'Total'],
      axisLine: { lineStyle: { color: alpha(theme.palette.divider, 0.2) } },
      axisTick: { show: false },
      axisLabel: { 
        color: theme.palette.text.secondary, 
        fontWeight: 800, 
        fontSize: 10,
        fontFamily: 'Outfit',
        margin: 15
      }
    },
    yAxis: {
      type: 'value',
      splitLine: { lineStyle: { color: alpha(theme.palette.divider, 0.05), type: 'dashed' } },
      axisLabel: { 
        color: theme.palette.text.secondary, 
        fontFamily: 'JetBrains Mono',
        fontSize: 10
      }
    },
    series: [
      {
        name: 'Offset',
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
        label: { 
          show: true, 
          position: 'top', 
          fontFamily: 'JetBrains Mono',
          fontWeight: 800,
          fontSize: 10,
          formatter: (p: any) => `$${p.value >= 0 ? '+' : ''}${p.value.toFixed(1)}` 
        },
        itemStyle: {
          borderRadius: 6,
          color: (p: any) => {
            if (p.name === 'Total') return qfd?.emerald ?? '#10b981';
            return p.value >= 0 ? qfd?.emerald ?? '#10b981' : theme.palette.error.main;
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
      p: 4, 
      height: '100%', 
      borderRadius: 8, 
      bgcolor: alpha(theme.palette.background.paper, 0.2),
      border: `1px solid ${alpha('#f8fafc', 0.05)}`,
      backdropFilter: 'blur(30px)',
      position: 'relative',
      overflow: 'hidden',
      '&::before': {
        content: '""',
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '4px',
        background: `linear-gradient(90deg, transparent, ${alpha(qfd?.emerald ?? '#10b981', 0.3)}, transparent)`,
      }
    }}>
      <Typography variant="caption" sx={{ 
        fontWeight: 900, 
        color: alpha(theme.palette.text.secondary, 0.6), 
        letterSpacing: '0.2em', 
        textTransform: 'uppercase', 
        mb: 3, 
        display: 'flex',
        alignItems: 'center',
        gap: 1.5
      }}>
        <Box sx={{ width: 4, height: 4, bgcolor: qfd?.emerald, borderRadius: '50%' }} />
        Portfolio Alpha Attribution
      </Typography>
      <ReactECharts option={option} style={{ height: '320px' }} />
    </Box>
  );
};
