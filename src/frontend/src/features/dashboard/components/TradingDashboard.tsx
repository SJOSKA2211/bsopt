import React, { useState, lazy, Suspense } from 'react';
import {
  Box,
  CssBaseline,
  AppBar,
  Toolbar,
  Typography,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  IconButton,
  Grid,
  Paper,
  Container,
  alpha,
  useTheme,
  CircularProgress,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  ShowChart as ChartIcon,
  AccountBalanceWallet as PortfolioIcon,
  Settings as SettingsIcon,
  AccountCircle,
} from '@mui/icons-material';
import { PortfolioSummary } from '../../portfolio/components/PortfolioSummary';
import { MLPredictions } from '../../options/components/MLPredictions';

// Lazy loaded heavy components
const OptionsChain = lazy(() => import('../../options/components/OptionsChain').then(m => ({ default: m.OptionsChain })));
const LivePriceChart = lazy(() => import('../../charts/components/LivePriceChart').then(m => ({ default: m.LivePriceChart })));
const GreeksHeatmap = lazy(() => import('../../options/components/GreeksHeatmap').then(m => ({ default: m.GreeksHeatmap })));
const VolatilitySurface3D = lazy(() => import('../../options/components/VolatilitySurface3D').then(m => ({ default: m.VolatilitySurface3D })));

const drawerWidth = 240;

const LoadingFallback: React.FC = () => (
  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
    <CircularProgress size={40} />
  </Box>
);

export const TradingDashboard: React.FC = () => {
  const theme = useTheme();
  const [drawerOpen, setDrawerOpen] = useState(true);

  const toggleDrawer = () => {
    setDrawerOpen(!drawerOpen);
  };

  return (
    <Box sx={{ display: 'flex', height: '100vh', bgcolor: 'background.default' }}>
      <CssBaseline />
      
      {/* Header */}
      <AppBar
        position="fixed"
        sx={{
          zIndex: theme.zIndex.drawer + 1,
          bgcolor: alpha(theme.palette.background.paper, 0.8),
          backdropFilter: 'blur(8px)',
          borderBottom: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
          boxShadow: 'none',
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={toggleDrawer}
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1, fontWeight: 'bold', color: 'primary.main' }}>
            BS-Opt Trading Dashboard
          </Typography>
          <IconButton color="inherit">
            <AccountCircle />
          </IconButton>
        </Toolbar>
      </AppBar>

      {/* Sidebar Navigation */}
      <Drawer
        variant="permanent"
        open={drawerOpen}
        sx={{
          width: drawerOpen ? drawerWidth : 72,
          flexShrink: 0,
          whiteSpace: 'nowrap',
          boxSizing: 'border-box',
          '& .MuiDrawer-paper': {
            width: drawerOpen ? drawerWidth : 72,
            overflowX: 'hidden',
            transition: theme.transitions.create('width', {
              easing: theme.transitions.easing.sharp,
              duration: theme.transitions.duration.enteringScreen,
            }),
            bgcolor: 'background.paper',
            borderRight: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
          },
        }}
        role="navigation"
      >
        <Toolbar />
        <Box sx={{ overflow: 'auto' }}>
          <List>
            {[
              { text: 'Dashboard', icon: <DashboardIcon /> },
              { text: 'Market Data', icon: <ChartIcon /> },
              { text: 'Portfolio', icon: <PortfolioIcon /> },
              { text: 'Settings', icon: <SettingsIcon /> },
            ].map((item) => (
              <ListItem key={item.text} disablePadding sx={{ display: 'block' }}>
                <ListItemButton
                  sx={{
                    minHeight: 48,
                    justifyContent: drawerOpen ? 'initial' : 'center',
                    px: 2.5,
                  }}
                >
                  <ListItemIcon
                    sx={{
                      minWidth: 0,
                      mr: drawerOpen ? 3 : 'auto',
                      justifyContent: 'center',
                      color: 'primary.main',
                    }}
                  >
                    {item.icon}
                  </ListItemIcon>
                  <ListItemText primary={item.text} sx={{ opacity: drawerOpen ? 1 : 0 }} />
                </ListItemButton>
              </ListItem>
            ))}
          </List>
        </Box>
      </Drawer>

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          height: '100vh',
          overflow: 'auto',
          pt: 8,
          pb: 2,
          px: 2,
        }}
      >
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
      </Box>
    </Box>
  );
};
