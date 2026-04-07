import React from 'react';
import ReactECharts from 'echarts-for-react';
import { alpha, Box } from '@mui/material';
import { stitchTokens } from '../../../theme/stitch-tokens';

export const EquityCurveChart: React.FC = () => {

  const option = {
    backgroundColor: 'transparent',
    grid: {
      left: '2%',
      right: '2%',
      bottom: '5%',
      top: '12%',
      containLabel: true
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(11, 14, 18, 0.95)',
      borderColor: stitchTokens.colors.primary,
      borderWidth: 1,
      padding: [12, 16],
      textStyle: { 
        color: '#fff', 
        fontFamily: 'JetBrains Mono', 
        fontSize: 10,
        fontWeight: 900
      },
      axisPointer: {
        lineStyle: {
          color: alpha(stitchTokens.colors.primary, 0.5),
          width: 1,
          type: 'dashed'
        }
      },
      formatter: (params: any) => {
        const val = params[0].value;
        return `[ TELEMETRY_SCAN ]<br/>VALUE: $${val.toLocaleString()}<br/>EPOCH: ${params[0].name}`;
      }
    },
    xAxis: {
      type: 'category',
      data: ['T-6', 'T-5', 'T-4', 'T-3', 'T-2', 'T-1', 'NOW'],
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } },
      axisLabel: { 
        color: 'rgba(255,255,255,0.3)', 
        fontSize: 8, 
        fontFamily: 'JetBrains Mono',
        fontWeight: 900,
        margin: 12
      },
      axisTick: { show: false }
    },
    yAxis: {
      type: 'value',
      axisLine: { show: false },
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.03)', type: 'dashed' } },
      axisLabel: { 
        color: 'rgba(255,255,255,0.3)', 
        fontSize: 8, 
        fontFamily: 'JetBrains Mono',
        fontWeight: 900,
        formatter: (v: number) => `$${(v/1000).toFixed(0)}K`
      }
    },
    series: [
      {
        data: [82000, 93200, 90100, 93400, 129000, 133000, 132042],
        type: 'line',
        smooth: 0.4,
        symbol: 'circle',
        symbolSize: 4,
        itemStyle: { color: stitchTokens.colors.primary },
        lineStyle: {
          width: 3,
          color: stitchTokens.colors.primary,
          shadowBlur: 20,
          shadowColor: alpha(stitchTokens.colors.primary, 0.5)
        },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: alpha(stitchTokens.colors.primary, 0.2) },
              { offset: 0.8, color: 'transparent' }
            ]
          }
        }
      }
    ]
  };

  return (
    <Box sx={{ position: 'relative', height: '100%', width: '100%' }}>
       <Box className="stitch-dots-container" sx={{ opacity: 0.02 }} />
       <ReactECharts option={option} style={{ height: '100%', width: '100%' }} />
    </Box>
  );
};
