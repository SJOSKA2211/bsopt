import React, { lazy, Suspense } from 'react';
import { Box, Container, Grid, Paper, Typography, CircularProgress } from '@mui/material';

const LivePriceChart = lazy(() => import('../../features/charts/components/LivePriceChart').then(m => ({ default: m.LivePriceChart })));
const GreeksHeatmap = lazy(() => import('../../features/options/components/GreeksHeatmap').then(m => ({ default: m.GreeksHeatmap })));
const OptionsChain = lazy(() => import('../../features/options/components/OptionsChain').then(m => ({ default: m.OptionsChain })));

const LoadingFallback: React.FC = () => (
  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', minHeight: 400 }}>
    <CircularProgress aria-label="Loading component" />
  </Box>
);

export const MarketPage: React.FC = () => {
  return (
    <Container maxWidth="xl" sx={{ mt: 2 }}>
      <Typography variant="h4" gutterBottom fontWeight="bold">Market Data</Typography>
      <Grid container spacing={3}>
        <Grid size={{ xs: 12 }}>
          <Paper sx={{ p: 2, height: 500 }}>
            <Suspense fallback={<LoadingFallback />}>
              <LivePriceChart symbol="AAPL" />
            </Suspense>
          </Paper>
        </Grid>
        <Grid size={{ xs: 12, lg: 8 }}>
            <Paper sx={{ p: 0, height: 600, overflow: 'hidden' }}>
                <Suspense fallback={<LoadingFallback />}>
                    <OptionsChain symbol="AAPL" />
                </Suspense>
            </Paper>
        </Grid>
        <Grid size={{ xs: 12, lg: 4 }}>
            <Paper sx={{ p: 2, height: 600 }}>
                <Suspense fallback={<LoadingFallback />}>
                    <GreeksHeatmap symbol="AAPL" greek="delta" />
                </Suspense>
            </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default MarketPage;
