import React from 'react';
import { Container, Typography, Paper } from '@mui/material';
import LazyLineChart from '@/components/charts/LazyLineChart';

const data = [
  { name: 'Page A', uv: 4000, pv: 2400, amt: 2400 },
  { name: 'Page B', uv: 3000, pv: 1398, amt: 2210 },
  { name: 'Page C', uv: 2000, pv: 9800, amt: 2290 },
  { name: 'Page D', uv: 2780, pv: 3908, amt: 2000 },
];

const Analysis: React.FC = () => {
  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Analysis
      </Typography>
      <Paper sx={{ p: 2, height: 400 }}>
        <LazyLineChart data={data} height="100%" />
      </Paper>
    </Container>
  );
};

export default Analysis;

