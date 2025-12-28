import React, { Suspense } from 'react';
import { lazy } from 'react'; // eslint-disable-line @typescript-eslint/no-unused-vars
import { Box, Skeleton } from '@mui/material';

// Create a wrapper component that uses recharts
import RechartsWrapper, { ChartDataPoint } from './RechartsWrapper';

interface LazyLineChartProps {
  data: ChartDataPoint[];
  height?: number | string;
}

const LazyLineChart: React.FC<LazyLineChartProps> = ({ data, height = 400 }) => {
  return (
    <Box sx={{ width: '100%', height }}>
      <Suspense fallback={<Skeleton variant="rectangular" width="100%" height={height} />}>
        <RechartsWrapper data={data} />
      </Suspense>
    </Box>
  );
};

export default LazyLineChart;