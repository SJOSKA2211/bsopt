import React, { lazy, Suspense } from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  Stack,
  alpha,
  useTheme,
  Avatar,
  List,
  ListItem,
  ListItemText,
  Chip,
  Button,
  Divider,
  CircularProgress,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  MoreHoriz,
  Apple as AppleIcon,
  Storefront as StoreIcon,
  Subscriptions as SubsIcon,
  Person as PersonIcon,
  ShowChart as ChartIcon,
  Layers as LayersIcon,
} from '@mui/icons-material';

// Lazy loaded trading components
const LivePriceChart = lazy(() => import('../../features/charts/components/LivePriceChart').then(m => ({ default: m.LivePriceChart })));
const MLPredictions = lazy(() => import('../../features/options/components/MLPredictions').then(m => ({ default: m.MLPredictions })));
const PortfolioSummary = lazy(() => import('../../features/portfolio/components/PortfolioSummary').then(m => ({ default: m.PortfolioSummary })));
const OptionsChain = lazy(() => import('../../features/options/components/OptionsChain').then(m => ({ default: m.OptionsChain })));
const GreeksHeatmap = lazy(() => import('../../features/options/components/GreeksHeatmap').then(m => ({ default: m.GreeksHeatmap })));
const VolatilitySurface3D = lazy(() => import('../../features/options/components/VolatilitySurface3D').then(m => ({ default: m.VolatilitySurface3D })));

const LoadingFallback = () => (
  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', p: 4 }}>
    <CircularProgress size={30} aria-label="Loading component" />
  </Box>
);

// Mock components for mini charts
const MiniBarChart = () => (
  <Stack direction="row" alignItems="flex-end" spacing={1} sx={{ height: 40, mt: 2 }}>
    {[35, 60, 45, 80, 25, 50].map((h, i) => (
      <Box 
        key={i} 
        sx={{ 
          width: 8, 
          height: `${h}%`, 
          bgcolor: i === 3 ? 'primary.main' : alpha('#94a3b8', 0.2), 
          borderRadius: 0.5 
        }} 
      />
    ))}
  </Stack>
);

const MiniLineChart = ({ color }: { color: string }) => (
  <Box sx={{ height: 40, mt: 2, position: 'relative', overflow: 'hidden' }}>
    <svg width="100%" height="100%" viewBox="0 0 100 40" preserveAspectRatio="none">
      <path
        d="M0 30 Q 25 10, 50 25 T 100 15"
        fill="none"
        stroke={color}
        strokeWidth="2"
      />
    </svg>
  </Box>
);

export const DashboardPage: React.FC = () => {
  const theme = useTheme();

  const transactions = [
    { id: 1, label: 'Simon Pegg', date: 'Jul 28, 6:22 PM', amount: 44.00, type: 'Transfer', icon: <PersonIcon />, color: '#10b981' },
    { id: 2, label: 'Apple Music', date: 'Jul 26, 12:30 PM', amount: -9.99, type: 'Subscription', icon: <AppleIcon />, color: '#f43f5e' },
    { id: 3, label: '7-Eleven', date: 'Jul 19, 2:56 PM', amount: -5.18, type: 'Grocery Store', icon: <StoreIcon />, color: '#f59e0b' },
    { id: 4, label: 'Joe Davis', date: 'Jul 19, 1:23 PM', amount: 13.00, type: 'Transfer', icon: <PersonIcon />, color: '#10b981' },
    { id: 5, label: 'Framer', date: 'Jul 19, 10:08 AM', amount: -14.99, type: 'Subscription', icon: <SubsIcon />, color: '#a855f7' },
  ];

  return (
    <Box sx={{ maxWidth: 1400, mx: 'auto', pb: 8 }}>
      {/* Top Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Stack direction="row" justifyContent="space-between" alignItems="flex-start">
              <Box>
                <Typography variant="caption" sx={{ color: 'text.disabled', fontWeight: 600 }}>
                  TOTAL SPENDINGS
                </Typography>
                <Typography variant="h2" sx={{ fontWeight: 700, my: 0.5 }}>
                  $832.80 <Chip label="-12%" size="small" color="error" sx={{ ml: 1, height: 20, fontSize: 10 }} />
                </Typography>
              </Box>
              <Typography variant="caption" sx={{ color: 'text.disabled', display: 'flex', alignItems: 'center' }}>
                THIS WEEK <MoreHoriz sx={{ fontSize: 16, ml: 0.5 }} />
              </Typography>
            </Stack>
            <MiniBarChart />
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Stack direction="row" justifyContent="space-between" alignItems="flex-start">
              <Box>
                <Typography variant="caption" sx={{ color: 'text.disabled', fontWeight: 600 }}>
                  SAVINGS
                </Typography>
                <Typography variant="h2" sx={{ fontWeight: 700, my: 0.5 }}>
                  $2,512.40 <Chip label="-2%" size="small" color="error" sx={{ ml: 1, height: 20, fontSize: 10 }} />
                </Typography>
              </Box>
              <Typography variant="caption" sx={{ color: 'text.disabled', display: 'flex', alignItems: 'center' }}>
                THIS YEAR <MoreHoriz sx={{ fontSize: 16, ml: 0.5 }} />
              </Typography>
            </Stack>
            <MiniLineChart color={theme.palette.error.main} />
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Stack direction="row" justifyContent="space-between" alignItems="flex-start">
              <Box>
                <Typography variant="caption" sx={{ color: 'text.disabled', fontWeight: 600 }}>
                  INVESTMENTS
                </Typography>
                <Typography variant="h2" sx={{ fontWeight: 700, my: 0.5 }}>
                  $1,215.25 <Chip label="+4%" size="small" color="success" sx={{ ml: 1, height: 20, fontSize: 10 }} />
                </Typography>
              </Box>
              <Typography variant="caption" sx={{ color: 'text.disabled', display: 'flex', alignItems: 'center' }}>
                THIS YEAR <MoreHoriz sx={{ fontSize: 16, ml: 0.5 }} />
              </Typography>
            </Stack>
            <MiniLineChart color={theme.palette.success.main} />
          </Paper>
        </Grid>
      </Grid>

      {/* Main Content Area */}
      <Grid container spacing={3} sx={{ mb: 6 }}>
        {/* Left Column: Transactions */}
        <Grid item xs={12} lg={4}>
          <Typography variant="h3" sx={{ mb: 2, fontWeight: 700 }}>
            Transactions
          </Typography>
          <Paper sx={{ p: 0, overflow: 'hidden' }}>
            <List disablePadding>
              {transactions.map((tx, i) => (
                <ListItem 
                  key={tx.id} 
                  sx={{ 
                    py: 2, 
                    px: 3, 
                    display: 'block',
                    borderBottom: i < transactions.length - 1 ? `1px solid ${alpha(theme.palette.divider, 0.1)}` : 'none'
                  }}
                >
                  <Stack direction="row" spacing={2} alignItems="center" sx={{ width: '100%' }}>
                    <Avatar 
                      sx={{ 
                        bgcolor: alpha(tx.color, 0.1), 
                        color: tx.color,
                        width: 40,
                        height: 40
                      }}
                    >
                      {tx.icon}
                    </Avatar>
                    <Box sx={{ flexGrow: 1 }}>
                      <Typography variant="body2" sx={{ fontWeight: 700 }}>
                        {tx.label}
                      </Typography>
                      <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                        {tx.date}
                      </Typography>
                    </Box>
                    <Box sx={{ textAlign: 'right' }}>
                      <Typography variant="body2" sx={{ fontWeight: 700, color: tx.amount > 0 ? 'success.main' : 'text.primary' }}>
                        {tx.amount > 0 ? '+' : ''}${Math.abs(tx.amount).toFixed(2)}
                      </Typography>
                      <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                        {tx.type}
                      </Typography>
                    </Box>
                  </Stack>
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>

        {/* Right Column: Analytics Chart */}
        <Grid item xs={12} lg={8}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
            <Typography variant="h2" sx={{ fontWeight: 700 }}>
              $9,340.80 <Typography component="span" variant="body1" sx={{ color: 'text.disabled', ml: 1 }}>Spent</Typography>
            </Typography>
            <Stack direction="row" spacing={1}>
              {['Day', 'Week', 'Month', 'Year'].map((t) => (
                <Button 
                  key={t} 
                  size="small" 
                  variant={t === 'Year' ? 'contained' : 'text'}
                  sx={{ 
                    minWidth: 60, 
                    color: t === 'Year' ? 'white' : 'text.disabled',
                    bgcolor: t === 'Year' ? alpha(theme.palette.text.primary, 0.1) : 'transparent',
                    '&:hover': { bgcolor: alpha(theme.palette.text.primary, 0.05) }
                  }}
                >
                  {t}
                </Button>
              ))}
            </Stack>
          </Stack>
          
          <Paper 
            sx={{ 
              height: 440, 
              p: 4, 
              display: 'flex', 
              flexDirection: 'column',
              justifyContent: 'flex-end',
              position: 'relative',
              overflow: 'hidden'
            }}
          >
            {/* Mock Stacked Bar Chart */}
            <Stack direction="row" spacing={4} alignItems="flex-end" sx={{ height: '100%', px: 2, zIndex: 0 }}>
              {['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL'].map((month, i) => (
                <Stack key={month} spacing={0.5} sx={{ flex: 1, alignItems: 'center' }}>
                  <Box sx={{ width: 12, display: 'flex', flexDirection: 'column-reverse', height: 250 + Math.random() * 80 }}>
                    <Box sx={{ height: '20%', bgcolor: 'primary.main', borderRadius: '0 0 4px 4px' }} />
                    <Box sx={{ height: '15%', bgcolor: 'secondary.main' }} />
                    <Box sx={{ height: '25%', bgcolor: 'warning.main' }} />
                    <Box sx={{ height: '40%', bgcolor: 'financial.accents.violet', borderRadius: '4px 4px 0 0' }} />
                  </Box>
                  <Typography variant="caption" sx={{ color: 'text.disabled', fontWeight: 600, mt: 1 }}>
                    {month}
                  </Typography>
                </Stack>
              ))}
            </Stack>
            
            {/* Tooltip Overlay Mock */}
            <Paper 
              sx={{ 
                position: 'absolute', 
                top: '40%', 
                left: '55%', 
                p: 1.5, 
                bgcolor: 'background.elevation2',
                border: `1px solid ${theme.palette.divider}`,
                zIndex: 1,
                boxShadow: '0 10px 30px rgba(0,0,0,0.5)'
              }}
            >
              <Typography variant="caption" sx={{ display: 'block', color: 'text.disabled' }}>TOTAL SPENT</Typography>
              <Typography variant="body1" sx={{ fontWeight: 700 }}>$589.40</Typography>
            </Paper>
          </Paper>
        </Grid>
      </Grid>

      {/* Trading Integration Section */}
      <Typography variant="h3" sx={{ mb: 3, fontWeight: 700, display: 'flex', alignItems: 'center' }}>
        <ChartIcon sx={{ mr: 1.5, color: 'primary.main' }} /> Trading Overview
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} lg={8}>
          <Paper data-testid="live-price-chart-paper" sx={{ p: 3, height: 500 }}>
            <Typography variant="h4" sx={{ mb: 2, fontWeight: 600 }}>Real-Time Market - AAPL</Typography>
            <Box sx={{ height: 400 }}>
              <Suspense fallback={<LoadingFallback />}>
                <LivePriceChart symbol="AAPL" />
              </Suspense>
            </Box>
          </Paper>
        </Grid>
        
        <Grid item xs={12} lg={4}>
          <Stack spacing={3}>
            <Paper data-testid="ml-predictions-paper" sx={{ p: 0, height: 235, overflow: 'hidden' }}>
              <Suspense fallback={<LoadingFallback />}>
                <MLPredictions symbol="AAPL" />
              </Suspense>
            </Paper>
            <Paper data-testid="portfolio-summary-container" sx={{ p: 0, height: 235, overflow: 'hidden' }}>
              <Suspense fallback={<LoadingFallback />}>
                <PortfolioSummary />
              </Suspense>
            </Paper>
          </Stack>
        </Grid>

        <Grid item xs={12} lg={8}>
          <Paper data-testid="options-chain-container" sx={{ p: 0, height: 600, overflow: 'hidden' }}>
            <Typography variant="h6" sx={{ p: 3, pb: 0, fontWeight: 600 }}>Options Chain</Typography>
            <Suspense fallback={<LoadingFallback />}>
              <OptionsChain symbol="AAPL" />
            </Suspense>
          </Paper>
        </Grid>

        <Grid item xs={12} lg={4}>
          <Stack spacing={3}>
            <Paper data-testid="greeks-heatmap-paper" sx={{ p: 3, height: 285 }}>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Greeks Delta</Typography>
              <Suspense fallback={<LoadingFallback />}>
                <GreeksHeatmap symbol="AAPL" greek="delta" />
              </Suspense>
            </Paper>
            <Paper data-testid="volatility-surface-paper" sx={{ p: 3, height: 285 }}>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Volatility Surface</Typography>
              <Suspense fallback={<LoadingFallback />}>
                <VolatilitySurface3D symbol="AAPL" />
              </Suspense>
            </Paper>
          </Stack>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DashboardPage;
