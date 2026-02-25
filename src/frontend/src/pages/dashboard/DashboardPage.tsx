import React, { lazy, Suspense } from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  Container,
  CircularProgress,
  List,
  ListItem,
  ListItemText,
  Divider,
  Chip,
  Stack,
  // useTheme,
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
  // const theme = useTheme();

  const transactions = [
    { id: 1, label: 'Apple Music', category: 'Subscription', amount: -9.99, time: '08:32 AM' },
    { id: 2, label: '7‑Eleven', category: 'Groceries', amount: -35.18, time: 'Yesterday' },
    { id: 3, label: 'Notion', category: 'Subscription', amount: -17.99, time: 'Jul 18' },
    { id: 4, label: 'Incoming Transfer', category: 'Transfer', amount: 215.5, time: 'Jul 16' },
  ];

  return (
    <Box
      sx={{
        minHeight: '100vh',
        py: 3,
        px: 2,
        background:
          'radial-gradient(circle at 15% 20%, rgba(16, 185, 129, 0.15), transparent 55%), radial-gradient(circle at 85% 10%, rgba(56, 189, 248, 0.18), transparent 55%), radial-gradient(circle at 50% 100%, rgba(236, 72, 153, 0.18), transparent 60%)',
        bgcolor: '#020617',
      }}
    >
      <Container maxWidth="xl">
        <Grid container spacing={3}>
          {/* Left rail: Summary + Transactions (like reference UI) */}
          <Grid size={{ xs: 12, md: 4, lg: 3 }} className="slide-up" style={{ animationDelay: '0.05s' }}>
            <Stack spacing={2}>
              <Paper
                sx={{
                  p: 2.5,
                  borderRadius: 3,
                  bgcolor: 'rgba(15,23,42,0.9)',
                  border: '1px solid rgba(148,163,184,0.35)',
                  boxShadow: '0 18px 45px rgba(0,0,0,0.55)',
                  color: '#e5e7eb',
                }}
              >
                <Typography variant="subtitle2" sx={{ mb: 1, color: 'rgba(148,163,184,0.9)' }}>
                  Total P&L (YTD)
                </Typography>
                <Typography variant="h5" sx={{ fontWeight: 700 }}>
                  $9,340.80
                </Typography>
                <Typography variant="caption" sx={{ color: '#22c55e' }}>
                  +12.4% vs last year
                </Typography>
              </Paper>

              <Paper
                sx={{
                  p: 0,
                  borderRadius: 3,
                  bgcolor: 'rgba(15,23,42,0.92)',
                  border: '1px solid rgba(30,64,175,0.55)',
                  boxShadow: '0 18px 45px rgba(0,0,0,0.6)',
                  color: '#e5e7eb',
                  overflow: 'hidden',
                }}
              >
                <Box sx={{ px: 2.5, pt: 2, pb: 1.5 }}>
                  <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
                    Transactions
                  </Typography>
                  <Typography variant="caption" sx={{ color: 'rgba(148,163,184,0.9)' }}>
                    Today & recent activity
                  </Typography>
                </Box>
                <Divider sx={{ borderColor: 'rgba(30,64,175,0.6)' }} />
                <List dense disablePadding>
                  {transactions.map((tx, index) => (
                    <React.Fragment key={tx.id}>
                      <ListItem
                        sx={{
                          px: 2.5,
                          py: 1.25,
                          '&:hover': {
                            bgcolor: 'rgba(15,23,42,0.9)',
                          },
                        }}
                      >
                        <ListItemText
                          primary={
                            <Stack direction="row" justifyContent="space-between" alignItems="center">
                              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                {tx.label}
                              </Typography>
                              <Typography
                                variant="body2"
                                sx={{ color: tx.amount >= 0 ? '#4ade80' : '#f97373', fontVariantNumeric: 'tabular-nums' }}
                              >
                                {tx.amount >= 0 ? '+' : '-'}${Math.abs(tx.amount).toFixed(2)}
                              </Typography>
                            </Stack>
                          }
                          secondary={
                            <Stack direction="row" justifyContent="space-between" alignItems="center">
                              <Typography variant="caption" sx={{ color: 'rgba(148,163,184,0.9)' }}>
                                {tx.time}
                              </Typography>
                              <Chip
                                label={tx.category}
                                size="small"
                                sx={{
                                  height: 20,
                                  fontSize: 10,
                                  borderRadius: 999,
                                  bgcolor: 'rgba(15,23,42,0.9)',
                                  border: '1px solid rgba(148,163,184,0.7)',
                                  color: 'rgba(209,213,219,0.9)',
                                }}
                              />
                            </Stack>
                          }
                        />
                      </ListItem>
                      {index < transactions.length - 1 && (
                        <Divider component="li" sx={{ borderColor: 'rgba(30,64,175,0.4)', ml: 2.5, mr: 2.5 }} />
                      )}
                    </React.Fragment>
                  ))}
                </List>
              </Paper>
            </Stack>
          </Grid>

          {/* Main analytics area */}
          <Grid size={{ xs: 12, md: 8, lg: 9 }} container spacing={3}>
        {/* Real-Time Price Chart */}
        <Grid size={{ xs: 12, lg: 8 }} className="slide-up" style={{ animationDelay: '0.1s' }}>
          <Paper
            data-testid="live-price-chart-paper"
            sx={{
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              height: 450,
              borderRadius: 3,
              bgcolor: 'rgba(15,23,42,0.96)',
              border: '1px solid rgba(148,163,184,0.45)',
              boxShadow: '0 22px 55px rgba(0,0,0,0.75)',
              color: '#e5e7eb',
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
        <Grid size={{ xs: 12, lg: 4 }} className="slide-up" style={{ animationDelay: '0.2s' }}>
          <Paper
            data-testid="ml-predictions-paper"
            sx={{
              p: 0,
              display: 'flex',
              flexDirection: 'column',
              height: 450,
              overflow: 'hidden',
              borderRadius: 3,
              bgcolor: 'rgba(15,23,42,0.96)',
              border: '1px solid rgba(148,163,184,0.45)',
              boxShadow: '0 22px 55px rgba(0,0,0,0.75)',
            }}
          >
            <MLPredictions symbol="AAPL" />
          </Paper>
        </Grid>

        {/* Options Chain Section */}
        <Grid size={{ xs: 12, lg: 8 }} className="slide-up" style={{ animationDelay: '0.3s' }}>
          <Paper
            data-testid="options-chain-container"
            sx={{
              p: 0,
              display: 'flex',
              flexDirection: 'column',
              height: 600,
              overflow: 'hidden',
              borderRadius: 3,
              bgcolor: 'rgba(15,23,42,0.96)',
              border: '1px solid rgba(30,64,175,0.6)',
              boxShadow: '0 22px 55px rgba(0,0,0,0.8)',
            }}
          >
            <Suspense fallback={<LoadingFallback />}>
              <OptionsChain symbol="AAPL" />
            </Suspense>
          </Paper>
        </Grid>

        {/* Portfolio Summary Section */}
        <Grid size={{ xs: 12, lg: 4 }} className="slide-up" style={{ animationDelay: '0.4s' }}>
          <Paper
            data-testid="portfolio-summary-container"
            sx={{
              p: 0,
              display: 'flex',
              flexDirection: 'column',
              height: 600,
              overflow: 'hidden',
              borderRadius: 3,
              bgcolor: 'rgba(15,23,42,0.96)',
              border: '1px solid rgba(30,64,175,0.6)',
              boxShadow: '0 22px 55px rgba(0,0,0,0.8)',
            }}
          >
            <PortfolioSummary />
          </Paper>
        </Grid>

        {/* Greeks Heatmap Summary */}
        <Grid size={{ xs: 12, lg: 4 }} className="slide-up" style={{ animationDelay: '0.5s' }}>
          <Paper
            data-testid="greeks-heatmap-paper"
            sx={{
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              height: 450,
              borderRadius: 3,
              bgcolor: 'rgba(15,23,42,0.96)',
              border: '1px solid rgba(15,23,42,0.9)',
              boxShadow: '0 22px 55px rgba(0,0,0,0.75)',
              color: '#e5e7eb',
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
        <Grid size={{ xs: 12, lg: 8 }} className="slide-up" style={{ animationDelay: '0.6s' }}>
          <Paper
            data-testid="volatility-surface-paper"
            sx={{
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              height: 450,
              borderRadius: 3,
              bgcolor: 'rgba(15,23,42,0.96)',
              border: '1px solid rgba(15,23,42,0.9)',
              boxShadow: '0 22px 55px rgba(0,0,0,0.75)',
              color: '#e5e7eb',
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
        </Grid>
      </Container>
    </Box>
  );
};

export default DashboardPage;
