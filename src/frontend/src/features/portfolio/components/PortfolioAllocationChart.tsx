import React from 'react';
import ReactECharts from 'echarts-for-react';
import { useTheme } from '@mui/material';

export const PortfolioAllocationChart: React.FC = () => {
  const theme = useTheme();

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      backgroundColor: 'rgba(15, 23, 42, 0.9)',
      borderColor: theme.palette.primary.main,
      textStyle: { color: '#fff' }
    },
    legend: {
      orient: 'vertical',
      left: 'left',
      textStyle: { color: theme.palette.text.secondary }
    },
    series: [
      {
        name: 'Allocation',
        type: 'pie',
        radius: ['55%', '85%'],
        avoidLabelOverlap: false,
        itemStyle: {
          borderRadius: 4,
          borderColor: '#0a0b14',
          borderWidth: 2
        },
        label: { show: false },
        emphasis: {
          label: {
            show: true,
            fontSize: '14',
            fontWeight: 'bold',
            color: '#fff'
          }
        },
        labelLine: { show: false },
        data: [
          { value: 1048, name: 'Technology', itemStyle: { color: '#00ffa3' } },
          { value: 735, name: 'Healthcare', itemStyle: { color: '#ff2e7e' } },
          { value: 580, name: 'Financials', itemStyle: { color: '#00d4ff' } },
          { value: 484, name: 'Consumer', itemStyle: { color: '#a855f7' } },
          { value: 300, name: 'Energy', itemStyle: { color: '#f59e0b' } }
        ]
      }
    ]
  };

  return <ReactECharts option={option} style={{ height: '300px', width: '100%' }} />;
};
