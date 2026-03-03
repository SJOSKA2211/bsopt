import React, { lazy, Suspense, useState } from 'react';
import {
  Box,
  Container,
  Grid,
  Paper,
  Typography,
  CircularProgress,
  Stack,
  Chip,
  Tab,
  Tabs,
  alpha,
  useTheme,
} from '@mui/material';
import {
  TrendingFlat as IVIcon,
  BarChart as HVIcon,
  SwapVert as PCRIcon,
} from '@mui/icons-material';

// Lazy loaded trading components
const LivePriceChart = lazy(() =>
  import('../../features/charts/components/LivePriceChart').then(m => ({ default: m.LivePriceChart }))
);
const GreeksHeatmap = lazy(() =>
  import('../../features/options/components/GreeksHeatmap').then(m => ({ default: m.GreeksHeatmap }))
);
const OptionsChain = lazy(() =>
  import('../../features/options/components/OptionsChain').then(m => ({ default: m.OptionsChain }))
);

const LoadingFallback: React.FC = () => (
  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', minHeight: 300 }}>
    <CircularProgress aria-label="Loading component" />
  </Box>
);

const SYMBOLS = ['AAPL', 'SPY', 'QQQ', 'NVDA', 'TSLA'];

const MARKET_PULSE = [
  { label: 'IV Rank', value: '42.3%', color: '#fbbf24', icon: <IVIcon sx={{ fontSize: 14 }} /> },
  { label: 'HV30', value: '28.1%', color: '#38bdf8', icon: <HVIcon sx={{ fontSize: 14 }} /> },
  { label: 'P/C Ratio', value: '0.87', color: '#10b981', icon: <PCRIcon sx={{ fontSize: 14 }} /> },
  { label: 'Volume', value: '73.2M', color: '#f8fafc', icon: null },
  { label: 'Open Int.', value: '4.82B', color: '#f8fafc', icon: null },
];

export const MarketPage: React.FC = () => {
  const theme = useTheme();
  const [symbol, setSymbol] = useState(0);

  const currentSymbol = SYMBOLS[symbol];

  return (
    <Container maxWidth="xl" sx={{ mt: 2, pb: 6 }}>
      {/* Header */}
      <Stack direction="row" justifyContent="space-between" alignItems="flex-end" sx={{ mb: 3 }}>
        <Box>
          <Typography
            variant="h3"
            className="text-gradient slide-up"
            sx={{ fontWeight: 800, mb: 0.5 }}
          >
            Market Data
          </Typography>
          <Typography variant="body2" sx={{ color: 'text.disabled' }}>
            Real-time options analytics & volatility surface
          </Typography>
        </Box>
        <Chip
          icon={<Box component="span" className="live-dot" />}
          label="STREAMING"
          size="small"
          sx={{
            bgcolor: alpha('#10b981', 0.1),
            color: 'success.main',
            border: `1px solid ${alpha('#10b981', 0.2)}`,
            fontWeight: 700,
            fontSize: '0.65rem',
            letterSpacing: '0.07em',
            '& .MuiChip-icon': { ml: 1, mr: -0.5, width: 'auto', height: 'auto' },
          }}
        />
      </Stack>

      {/* Symbol tabs */}
      <Tabs
        value={symbol}
        onChange={(_, v) => setSymbol(v)}
        className="slide-up"
        sx={{
          mb: 2,
          '& .MuiTab-root': {
            minWidth: 72,
            fontWeight: 700,
            fontSize: '0.85rem',
            letterSpacing: '0.05em',
            color: 'text.disabled',
            textTransform: 'none',
          },
          '& .Mui-selected': { color: 'primary.main !important' },
          '& .MuiTabs-indicator': {
            background: 'linear-gradient(90deg, #10b981, #38bdf8)',
            borderRadius: 3,
            height: 3,
            boxShadow: `0 0 8px ${alpha('#10b981', 0.6)}`,
          },
        }}
      >
        {SYMBOLS.map(s => (
          <Tab key={s} label={s} />
        ))}
      </Tabs>

      {/* Market Pulse strip */}
      <Paper
        className="slide-up"
        sx={{
          p: 0,
          mb: 3,
          borderRadius: 3,
          overflow: 'hidden',
        }}
      >
        <Stack
          direction="row"
          divider={
            <Box sx={{ width: 1, bgcolor: alpha('#94a3b8', 0.08) }} />
          }
        >
          {MARKET_PULSE.map((stat) => (
            <Box
              key={stat.label}
              sx={{
                flex: 1,
                py: 1.75,
                px: 2.5,
                textAlign: 'center',
                transition: 'background 0.15s ease',
                '&:hover': { bgcolor: alpha('#94a3b8', 0.04) },
              }}
            >
              <Stack direction="row" spacing={0.5} justifyContent="center" alignItems="center" sx={{ mb: 0.25 }}>
                {stat.icon && (
                  <Box sx={{ color: stat.color, display: 'flex' }}>{stat.icon}</Box>
                )}
                <Typography
                  variant="caption"
                  sx={{ color: 'text.disabled', fontWeight: 700, letterSpacing: '0.08em', fontSize: '0.62rem' }}
                >
                  {stat.label}
                </Typography>
              </Stack>
              <Typography
                variant="body2"
                sx={{
                  fontWeight: 700,
                  color: stat.color,
                  fontFamily: '"JetBrains Mono", monospace',
                  fontSize: '1rem',
                }}
              >
                {stat.value}
              </Typography>
            </Box>
          ))}
        </Stack>
      </Paper>

      {/* Charts */}
      <Grid container spacing={3}>
        <Grid size={{ xs: 12 }} className="slide-up" style={{ animationDelay: '0.1s' }}>
          <Paper sx={{ p: 2.5, height: 520 }}>
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
              <Stack direction="row" spacing={1.5} alignItems="baseline">
                <Typography variant="h5" sx={{ fontWeight: 700 }}>
                  {currentSymbol}
                </Typography>
                <Typography
                  sx={{ fontFamily: '"JetBrains Mono", monospace', fontWeight: 700, fontSize: '1.25rem', color: 'text.primary' }}
                >
                  $189.42
                </Typography>
                <Chip
                  label="▲ $2.18 (1.17%)"
                  size="small"
                  color="success"
                  sx={{ height: 22, fontSize: '0.7rem', fontWeight: 700 }}
                />
              </Stack>
            </Stack>
            <Box sx={{ height: 440 }}>
              <Suspense fallback={<LoadingFallback />}>
                <LivePriceChart symbol={currentSymbol} />
              </Suspense>
            </Box>
          </Paper>
        </Grid>

        <Grid size={{ xs: 12, lg: 8 }} className="slide-up" style={{ animationDelay: '0.2s' }}>
          <Paper sx={{ height: 600, overflow: 'hidden' }}>
            <Stack direction="row" spacing={1.5} alignItems="center" sx={{ p: 2.5, pb: 1 }}>
              <Typography variant="h5" sx={{ fontWeight: 700 }}>
                Options Chain
              </Typography>
              <Chip
                label={currentSymbol}
                size="small"
                sx={{
                  height: 20,
                  fontSize: '0.65rem',
                  fontWeight: 700,
                  bgcolor: alpha(theme.palette.primary.main, 0.1),
                  color: 'primary.main',
                }}
              />
            </Stack>
            <Suspense fallback={<LoadingFallback />}>
              <OptionsChain symbol={currentSymbol} />
            </Suspense>
          </Paper>
        </Grid>

        <Grid size={{ xs: 12, lg: 4 }} className="slide-up" style={{ animationDelay: '0.3s' }}>
          <Paper sx={{ p: 2.5, height: 600 }}>
            <Typography variant="h5" sx={{ fontWeight: 700, mb: 2 }}>
              Greeks · Delta Heatmap
            </Typography>
            <Suspense fallback={<LoadingFallback />}>
              <GreeksHeatmap symbol={currentSymbol} greek="delta" />
            </Suspense>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default MarketPage;
