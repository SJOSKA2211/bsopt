import React from 'react';
import ReactECharts from 'echarts-for-react';
import { useTheme, alpha } from '@mui/material';

export const EquityCurveChart: React.FC = () => {
  const theme = useTheme();

  const option = {
    backgroundColor: 'transparent',
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      top: '10%',
      containLabel: true
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(15, 23, 42, 0.9)',
      borderColor: theme.palette.primary.main,
      textStyle: { color: '#fff' }
    },
    xAxis: {
      type: 'category',
      data: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
      axisLine: { lineStyle: { color: alpha('#fff', 0.1) } },
      axisLabel: { color: theme.palette.text.secondary }
    },
    yAxis: {
      type: 'value',
      axisLine: { show: false },
      splitLine: { lineStyle: { color: alpha('#fff', 0.05) } },
      axisLabel: { color: theme.palette.text.secondary }
    },
    series: [
      {
        data: [820, 932, 901, 934, 1290, 1330, 1320],
        type: 'line',
        smooth: true,
        symbol: 'none',
        lineStyle: {
          width: 3,
          color: theme.palette.primary.main,
          shadowBlur: 10,
          shadowColor: theme.palette.primary.main
        },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: alpha(theme.palette.primary.main, 0.3) },
              { offset: 1, color: 'transparent' }
            ]
          }
        }
      }
    ]
  };

  return <ReactECharts option={option} style={{ height: '300px', width: '100%' }} />;
};
