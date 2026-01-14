import React, { lazy, Suspense } from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  Container,
  CircularProgress,
} from '@mui/material';
import { PortfolioSummary } from '../../features/portfolio/components/PortfolioSummary';
import { MLPredictions } from '../../features/options/components/MLPredictions';

// Lazy loaded heavy components
const OptionsChain = lazy(() => import('../../features/options/components/OptionsChain').then(m => ({ default: m.OptionsChain })));
const LivePriceChart = lazy(() => import('../../features/charts/components/LivePriceChart').then(m => ({ default: m.LivePriceChart })));
const GreeksHeatmap = lazy(() => import('../../features/options/components/GreeksHeatmap').then(m => ({ default: m.GreeksHeatmap })));
const VolatilitySurface3D = lazy(() => import('../../features/options/components/VolatilitySurface3D').then(m => ({ default: m.VolatilitySurface3D })));

const LoadingFallback: React.FC = () => (
  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
    <CircularProgress size={40} aria-label="Loading component" />
  </Box>
);

export const DashboardPage: React.FC = () => {
  return (
    <Container maxWidth="xl" sx={{ mt: 2, mb: 2 }}>
      <Grid container spacing={3}>
        {/* Real-Time Price Chart */}
        <Grid size={{ xs: 12, lg: 8 }}>
          <Paper
            data-testid="live-price-chart-paper"
            sx={{
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              height: 450,
            }}
          >
            <Typography variant="h6" gutterBottom>
              Real-Time Price Chart - AAPL
            </Typography>
            <Box sx={{ flex: 1, overflow: 'hidden' }}>
              <Suspense fallback={<LoadingFallback />}>
                <LivePriceChart symbol="AAPL" />
              </Suspense>
            </Box>
          </Paper>
        </Grid>

        {/* ML Predictions Widget */}
        <Grid size={{ xs: 12, lg: 4 }}>
          <Paper
            data-testid="ml-predictions-paper"
            sx={{
              p: 0,
              display: 'flex',
              flexDirection: 'column',
              height: 450,
              overflow: 'hidden',
            }}
          >
            <MLPredictions symbol="AAPL" />
          </Paper>
        </Grid>

        {/* Options Chain Section */}
        <Grid size={{ xs: 12, lg: 8 }}>
          <Paper
            data-testid="options-chain-container"
            sx={{
              p: 0,
              display: 'flex',
              flexDirection: 'column',
              height: 600,
              overflow: 'hidden',
            }}
          >
            <Suspense fallback={<LoadingFallback />}>
              <OptionsChain symbol="AAPL" />
            </Suspense>
          </Paper>
        </Grid>

        {/* Portfolio Summary Section */}
        <Grid size={{ xs: 12, lg: 4 }}>
          <Paper
            data-testid="portfolio-summary-container"
            sx={{
              p: 0,
              display: 'flex',
              flexDirection: 'column',
              height: 600,
              overflow: 'hidden',
            }}
          >
            <PortfolioSummary />
          </Paper>
        </Grid>

        {/* Greeks Heatmap Summary */}
        <Grid size={{ xs: 12, lg: 4 }}>
          <Paper
            data-testid="greeks-heatmap-paper"
            sx={{
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              height: 450,
            }}
          >
            <Typography variant="h6" gutterBottom>
              Greeks Analysis (Delta)
            </Typography>
            <Box sx={{ flex: 1, overflow: 'hidden' }}>
              <Suspense fallback={<LoadingFallback />}>
                <GreeksHeatmap symbol="AAPL" greek="delta" />
              </Suspense>
            </Box>
          </Paper>
        </Grid>

        {/* 3D Volatility Surface */}
        <Grid size={{ xs: 12, lg: 8 }}>
          <Paper
            data-testid="volatility-surface-paper"
            sx={{
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              height: 450,
            }}
          >
            <Typography variant="h6" gutterBottom>
              Implied Volatility Surface
            </Typography>
            <Box sx={{ flex: 1, overflow: 'hidden' }}>
              <Suspense fallback={<LoadingFallback />}>
                <VolatilitySurface3D symbol="AAPL" />
              </Suspense>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default DashboardPage;
