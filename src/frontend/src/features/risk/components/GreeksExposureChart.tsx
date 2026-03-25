import React from 'react';
import ReactECharts from 'echarts-for-react';
import { useTheme, alpha } from '@mui/material';

export const GreeksExposureChart: React.FC = () => {
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
    xAxis: {
      type: 'value',
      axisLine: { show: false },
      splitLine: { lineStyle: { color: alpha('#fff', 0.05) } },
      axisLabel: { color: theme.palette.text.secondary }
    },
    yAxis: {
      type: 'category',
      data: ['Delta', 'Gamma', 'Theta', 'Vega'],
      axisLine: { lineStyle: { color: alpha('#fff', 0.1) } },
      axisLabel: { color: '#fff', fontWeight: 800 }
    },
    series: [
      {
        name: 'Exposure',
        type: 'bar',
        data: [
          { value: 850, itemStyle: { color: '#00ffa3' } },
          { value: 125, itemStyle: { color: '#00d4ff' } },
          { value: -242, itemStyle: { color: '#ff2e7e' } },
          { value: 415, itemStyle: { color: '#a855f7' } }
        ],
        label: {
          show: true,
          position: 'right',
          color: '#fff',
          fontWeight: 800,
          fontFamily: 'JetBrains Mono'
        },
        itemStyle: {
          borderRadius: [0, 4, 4, 0]
        }
      }
    ]
  };

  return <ReactECharts option={option} style={{ height: '240px', width: '100%' }} />;
};
