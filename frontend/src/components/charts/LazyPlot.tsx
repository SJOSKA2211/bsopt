import React, { Suspense, lazy } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import { Data, Layout } from 'plotly.js'; // Import Plotly types

// Dynamically import react-plotly.js
const Plot = lazy(() => import('react-plotly.js'));

interface LazyPlotProps {
  data: Data[];
  layout: Partial<Layout>;
  title?: string;
  width?: string | number;
  height?: string | number;
}

const LazyPlot: React.FC<LazyPlotProps> = ({ data, layout, title, width = '100%', height = 400 }) => {
  return (
    <Box sx={{ width, height, position: 'relative', minHeight: height }}>
      {title && (
        <Typography variant="h6" gutterBottom>
          {title}
        </Typography>
      )}
      <Suspense
        fallback={
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              height: '100%',
              bgcolor: 'rgba(0,0,0,0.05)',
              borderRadius: 1,
            }}
          >
            <CircularProgress size={24} />
            <Typography variant="body2" sx={{ ml: 2 }}>
              Loading Chart...
            </Typography>
          </Box>
        }
      >
        <Plot
          data={data}
          layout={{
            ...layout,
            autosize: true,
          }}
          useResizeHandler={true}
          style={{ width: '100%', height: '100%' }}
        />
      </Suspense>
    </Box>
  );
};

export default LazyPlot;
