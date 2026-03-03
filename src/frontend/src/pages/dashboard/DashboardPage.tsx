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
  Chip,
  Button,
  CircularProgress,
  LinearProgress,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  ShowChart as ChartIcon,
  WaterfallChart as GreeksIcon,
  Bolt as MLIcon,
  AccountBalance as PortfolioIcon,
  CallMade as CallIcon,
  CallReceived as PutIcon,
} from '@mui/icons-material';

// Lazy loaded trading components
const LivePriceChart = lazy(() =>
  import('../../features/charts/components/LivePriceChart').then(m => ({ default: m.LivePriceChart }))
);
const MLPredictions = lazy(() =>
  import('../../features/options/components/MLPredictions').then(m => ({ default: m.MLPredictions }))
);
const PortfolioSummary = lazy(() =>
  import('../../features/portfolio/components/PortfolioSummary').then(m => ({ default: m.PortfolioSummary }))
);
const OptionsChain = lazy(() =>
  import('../../features/options/components/OptionsChain').then(m => ({ default: m.OptionsChain }))
);
const GreeksHeatmap = lazy(() =>
  import('../../features/options/components/GreeksHeatmap').then(m => ({ default: m.GreeksHeatmap }))
);
const VolatilitySurface3D = lazy(() =>
  import('../../features/options/components/VolatilitySurface3D').then(m => ({
    default: m.VolatilitySurface3D,
  }))
);

const LoadingFallback = () => (
  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', p: 4 }}>
    <CircularProgress size={28} aria-label="Loading component" />
  </Box>
);

// ---------------------------------------------------------------------------
// KPI Card – trading-domain summary stat
// ---------------------------------------------------------------------------
interface KpiCardProps {
  label: string;
  value: string;
  subValue?: string;
  positive?: boolean;
  neutral?: boolean;
  icon: React.ReactNode;
  accentColor: string;
  progress?: number; // 0–100
}

const KpiCard: React.FC<KpiCardProps> = ({
  label,
  value,
  subValue,
  positive,
  neutral,
  icon,
  accentColor,
  progress,
}) => {
  const theme = useTheme();
  return (
    <Paper
      className="stat-card"
      sx={{
        p: 3,
        position: 'relative',
        overflow: 'hidden',
        height: '100%',
        border: `1px solid ${alpha(accentColor, 0.15)}`,
      }}
    >
      {/* Background glow */}
      <Box
        sx={{
          position: 'absolute',
          top: -30,
          right: -30,
          width: 100,
          height: 100,
          borderRadius: '50%',
          bgcolor: alpha(accentColor, 0.08),
          filter: 'blur(24px)',
          pointerEvents: 'none',
        }}
      />
      {/* Greek overlay character */}
      <Box className="greek-bg-overlay" sx={{ fontSize: 80, color: accentColor }}>
        {label === 'Portfolio P&L' ? 'Δ' : label === 'Options Premium' ? 'Θ' : 'Γ'}
      </Box>

      <Stack direction="row" justifyContent="space-between" alignItems="flex-start">
        <Box>
          <Typography
            variant="caption"
            sx={{ color: 'text.disabled', fontWeight: 700, letterSpacing: '0.1em' }}
          >
            {label}
          </Typography>
          <Typography
            variant="h3"
            sx={{ fontWeight: 800, my: 0.75, fontFamily: '"JetBrains Mono", monospace', fontSize: '1.6rem' }}
          >
            {value}
          </Typography>
          {subValue && (
            <Chip
              size="small"
              label={subValue}
              color={positive ? 'success' : neutral ? 'default' : 'error'}
              sx={{ height: 22, fontSize: '0.68rem', fontWeight: 700 }}
            />
          )}
          {progress !== undefined && (
            <Box sx={{ mt: 1.5 }}>
              <LinearProgress
                variant="determinate"
                value={progress}
                sx={{
                  height: 6,
                  borderRadius: 3,
                  bgcolor: alpha(accentColor, 0.12),
                  '& .MuiLinearProgress-bar': {
                    borderRadius: 3,
                    background: `linear-gradient(90deg, ${accentColor}, ${alpha(accentColor, 0.6)})`,
                  },
                }}
              />
              <Typography variant="caption" sx={{ color: 'text.disabled', mt: 0.5, display: 'block' }}>
                Win rate: {progress}%
              </Typography>
            </Box>
          )}
        </Box>
        <Avatar
          sx={{
            bgcolor: alpha(accentColor, 0.12),
            color: accentColor,
            width: 44,
            height: 44,
            border: `1px solid ${alpha(accentColor, 0.2)}`,
          }}
        >
          {icon}
        </Avatar>
      </Stack>
    </Paper>
  );
};

// ---------------------------------------------------------------------------
// Mini sparkline
// ---------------------------------------------------------------------------
const MiniSparkline: React.FC<{ color: string; up: boolean }> = ({ color, up }) => (
  <Box sx={{ height: 36, mt: 1.5, position: 'relative', overflow: 'hidden' }}>
    <svg width="100%" height="100%" viewBox="0 0 100 36" preserveAspectRatio="none">
      <defs>
        <linearGradient id={`grad-${color}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.35" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      {up ? (
        <>
          <path d="M0 30 L10 26 L25 18 L40 22 L55 12 L70 16 L85 6 L100 2" fill="none" stroke={color} strokeWidth="1.5" />
          <path d="M0 30 L10 26 L25 18 L40 22 L55 12 L70 16 L85 6 L100 2 L100 36 L0 36Z" fill={`url(#grad-${color})`} />
        </>
      ) : (
        <>
          <path d="M0 6 L10 10 L25 8 L40 16 L55 14 L70 22 L85 20 L100 28" fill="none" stroke={color} strokeWidth="1.5" />
          <path d="M0 6 L10 10 L25 8 L40 16 L55 14 L70 22 L85 20 L100 28 L100 36 L0 36Z" fill={`url(#grad-${color})`} />
        </>
      )}
    </svg>
  </Box>
);

// ---------------------------------------------------------------------------
// Recent trades mock
// ---------------------------------------------------------------------------
const RECENT_TRADES = [
  { id: 1, symbol: 'AAPL', type: 'CALL', strike: '$185', exp: '21 Mar', qty: 5, pnl: +840.0, dir: true },
  { id: 2, symbol: 'SPY', type: 'PUT', strike: '$470', exp: '21 Mar', qty: 10, pnl: -320.0, dir: false },
  { id: 3, symbol: 'QQQ', type: 'CALL', strike: '$400', exp: '18 Apr', qty: 8, pnl: +1120.0, dir: true },
  { id: 4, symbol: 'NVDA', type: 'CALL', strike: '$480', exp: '28 Mar', qty: 3, pnl: +2640.0, dir: true },
  { id: 5, symbol: 'TSLA', type: 'PUT', strike: '$250', exp: '18 Apr', qty: 6, pnl: -180.0, dir: false },
];

const TIME_FILTERS = ['1D', '1W', '1M', '3M', '1Y'];

// ---------------------------------------------------------------------------
// Dashboard Page
// ---------------------------------------------------------------------------
export const DashboardPage: React.FC = () => {
  const theme = useTheme();
  const [activeTime, setActiveTime] = React.useState('1M');

  return (
    <Box sx={{ maxWidth: 1400, mx: 'auto', pb: 8 }}>
      {/* ---- Page header ---- */}
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 4 }}>
        <Box>
          <Typography
            variant="h2"
            className="slide-up"
            sx={{ fontWeight: 800, mb: 0.5 }}
          >
            Good morning,{' '}
            <Box
              component="span"
              sx={{
                background: 'linear-gradient(135deg, #10b981, #38bdf8)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
              }}
            >
              Trader
            </Box>
          </Typography>
          <Typography variant="body2" sx={{ color: 'text.disabled' }}>
            Your quantitative options terminal — March 2, 2026
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<ChartIcon />}
          sx={{ px: 3, py: 1.25, fontSize: '0.9rem' }}
          onClick={() => { }}
        >
          Quick Trade
        </Button>
      </Stack>

      {/* ---- KPI Cards ---- */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid size={{ xs: 12, md: 4 }}>
          <KpiCard
            label="Portfolio P&L"
            value="+$12,340"
            subValue="+8.4% YTD"
            positive
            icon={<TrendingUpIcon />}
            accentColor={theme.palette.success.main}
          />
        </Grid>
        <Grid size={{ xs: 12, md: 4 }}>
          <KpiCard
            label="Options Premium"
            value="$4,892"
            subValue="-$210 today"
            icon={<GreeksIcon />}
            accentColor={theme.palette.secondary.main}
          />
        </Grid>
        <Grid size={{ xs: 12, md: 4 }}>
          <KpiCard
            label="Win Rate"
            value="73.2%"
            icon={<MLIcon />}
            accentColor={theme.palette.financial?.accents?.violet ?? '#a855f7'}
            progress={73}
          />
        </Grid>
      </Grid>

      {/* ---- Main content: Trades + Chart ---- */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {/* Recent trades */}
        <Grid size={{ xs: 12, lg: 4 }}>
          <Typography variant="h5" sx={{ mb: 2, fontWeight: 700 }}>
            Recent Trades
          </Typography>
          <Paper sx={{ overflow: 'hidden' }}>
            <List disablePadding>
              {RECENT_TRADES.map((trade, i) => (
                <ListItem
                  key={trade.id}
                  className={trade.dir ? 'trade-indicator-bullish' : 'trade-indicator-bearish'}
                  sx={{
                    py: 1.75,
                    px: 3,
                    display: 'block',
                    borderBottom:
                      i < RECENT_TRADES.length - 1
                        ? `1px solid ${alpha(theme.palette.divider, 0.5)}`
                        : 'none',
                    transition: 'background 0.15s ease',
                    '&:hover': { bgcolor: alpha('#94a3b8', 0.04) },
                  }}
                >
                  <Stack direction="row" spacing={2} alignItems="center">
                    <Avatar
                      sx={{
                        bgcolor: alpha(trade.dir ? '#10b981' : '#f43f5e', 0.1),
                        color: trade.dir ? 'success.main' : 'error.main',
                        width: 38,
                        height: 38,
                        border: `1px solid ${alpha(trade.dir ? '#10b981' : '#f43f5e', 0.2)}`,
                      }}
                    >
                      {trade.dir ? <CallIcon fontSize="small" /> : <PutIcon fontSize="small" />}
                    </Avatar>
                    <Box sx={{ flexGrow: 1 }}>
                      <Stack direction="row" spacing={1} alignItems="center">
                        <Typography variant="body2" sx={{ fontWeight: 700 }}>
                          {trade.symbol}
                        </Typography>
                        <Chip
                          label={trade.type}
                          size="small"
                          color={trade.dir ? 'success' : 'error'}
                          sx={{ height: 18, fontSize: '0.6rem', fontWeight: 700 }}
                        />
                      </Stack>
                      <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                        {trade.strike} · Exp {trade.exp} · Qty {trade.qty}
                      </Typography>
                    </Box>
                    <Box sx={{ textAlign: 'right' }}>
                      <Typography
                        variant="body2"
                        sx={{
                          fontWeight: 700,
                          fontFamily: '"JetBrains Mono", monospace',
                          color: trade.pnl > 0 ? 'success.main' : 'error.main',
                        }}
                      >
                        {trade.pnl > 0 ? '+' : ''}${Math.abs(trade.pnl).toFixed(2)}
                      </Typography>
                      <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                        P&L
                      </Typography>
                    </Box>
                  </Stack>
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>

        {/* Analytics chart */}
        <Grid size={{ xs: 12, lg: 8 }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
            <Box>
              <Typography variant="h3" sx={{ fontWeight: 800, fontFamily: '"JetBrains Mono", monospace' }}>
                $48,392
                <Typography
                  component="span"
                  variant="body1"
                  sx={{ color: 'text.disabled', ml: 1.5, fontFamily: 'Inter, sans-serif' }}
                >
                  Portfolio Value
                </Typography>
              </Typography>
              <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: 0.5 }}>
                <TrendingUpIcon sx={{ color: 'success.main', fontSize: 16 }} />
                <Typography variant="caption" sx={{ color: 'success.main', fontWeight: 600 }}>
                  +$3,240 this week
                </Typography>
              </Stack>
            </Box>
            <Stack direction="row" spacing={0.5}>
              {TIME_FILTERS.map((t) => (
                <Button
                  key={t}
                  size="small"
                  variant={t === activeTime ? 'contained' : 'text'}
                  onClick={() => setActiveTime(t)}
                  sx={{
                    minWidth: 44,
                    height: 32,
                    fontSize: '0.75rem',
                    color: t === activeTime ? 'white' : 'text.disabled',
                    ...(t !== activeTime && {
                      '&:hover': { color: 'text.primary', bgcolor: alpha('#94a3b8', 0.06) },
                    }),
                  }}
                >
                  {t}
                </Button>
              ))}
            </Stack>
          </Stack>

          <Paper sx={{ height: 400, p: 4 }}>
            {/* Stacked performance bar chart (aesthetic) */}
            <Stack
              direction="row"
              spacing={2}
              alignItems="flex-end"
              sx={{ height: '100%', px: 1 }}
            >
              {['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL'].map((month, idx) => {
                const total = 160 + Math.sin(idx * 0.9) * 60 + idx * 12;
                return (
                  <Stack key={month} spacing={0} sx={{ flex: 1, alignItems: 'center' }}>
                    <Box
                      sx={{
                        width: '100%',
                        maxWidth: 28,
                        display: 'flex',
                        flexDirection: 'column-reverse',
                        height: total,
                        borderRadius: '6px 6px 0 0',
                        overflow: 'hidden',
                        cursor: 'pointer',
                        transition: 'opacity 0.2s ease',
                        '&:hover': { opacity: 0.85 },
                      }}
                    >
                      <Box sx={{ height: '22%', bgcolor: 'primary.main' }} />
                      <Box sx={{ height: '18%', bgcolor: 'secondary.main' }} />
                      <Box sx={{ height: '28%', bgcolor: 'warning.main' }} />
                      <Box sx={{ height: '32%', bgcolor: alpha('#a855f7', 0.8) }} />
                    </Box>
                    <Typography
                      variant="caption"
                      sx={{ color: 'text.disabled', fontWeight: 600, mt: 1, fontSize: '0.65rem' }}
                    >
                      {month}
                    </Typography>
                  </Stack>
                );
              })}
            </Stack>
          </Paper>
        </Grid>
      </Grid>

      {/* ---- Trading widgets ---- */}
      <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 3 }}>
        <ChartIcon sx={{ color: 'primary.main' }} />
        <Typography variant="h4" sx={{ fontWeight: 700 }}>
          Trading Overview
        </Typography>
        <Chip
          label="LIVE"
          size="small"
          sx={{
            bgcolor: alpha('#10b981', 0.12),
            color: 'success.main',
            border: `1px solid ${alpha('#10b981', 0.25)}`,
            fontWeight: 700,
            fontSize: '0.6rem',
            letterSpacing: '0.08em',
            height: 20,
          }}
        />
      </Stack>

      <Grid container spacing={3}>
        {/* Main chart */}
        <Grid size={{ xs: 12, lg: 8 }}>
          <Paper data-testid="live-price-chart-paper" sx={{ p: 3, height: 500 }}>
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
              <Typography variant="h5" sx={{ fontWeight: 700 }}>
                Real-Time · AAPL
              </Typography>
              <Stack direction="row" spacing={0.5} alignItems="center">
                <Box className="live-dot" />
                <Typography variant="caption" sx={{ color: 'success.main', fontWeight: 600, fontSize: '0.7rem' }}>
                  LIVE
                </Typography>
              </Stack>
            </Stack>
            <Box sx={{ height: 400 }}>
              <Suspense fallback={<LoadingFallback />}>
                <LivePriceChart symbol="AAPL" />
              </Suspense>
            </Box>
          </Paper>
        </Grid>

        {/* Sidebar widgets */}
        <Grid size={{ xs: 12, lg: 4 }}>
          <Stack spacing={3}>
            <Paper data-testid="ml-predictions-paper" sx={{ height: 235, overflow: 'hidden' }}>
              <Stack direction="row" spacing={1} alignItems="center" sx={{ p: 2, pb: 0 }}>
                <MLIcon sx={{ color: 'secondary.main', fontSize: 18 }} />
                <Typography variant="body2" sx={{ fontWeight: 700 }}>
                  ML Predictions
                </Typography>
              </Stack>
              <Suspense fallback={<LoadingFallback />}>
                <MLPredictions symbol="AAPL" />
              </Suspense>
            </Paper>
            <Paper data-testid="portfolio-summary-container" sx={{ height: 235, overflow: 'hidden' }}>
              <Stack direction="row" spacing={1} alignItems="center" sx={{ p: 2, pb: 0 }}>
                <PortfolioIcon sx={{ color: 'primary.main', fontSize: 18 }} />
                <Typography variant="body2" sx={{ fontWeight: 700 }}>
                  Portfolio Summary
                </Typography>
              </Stack>
              <Suspense fallback={<LoadingFallback />}>
                <PortfolioSummary />
              </Suspense>
            </Paper>
          </Stack>
        </Grid>

        {/* Options chain */}
        <Grid size={{ xs: 12, lg: 8 }}>
          <Paper data-testid="options-chain-container" sx={{ height: 600, overflow: 'hidden' }}>
            <Stack direction="row" spacing={1} alignItems="center" sx={{ p: 3, pb: 1.5 }}>
              <ChartIcon sx={{ color: 'primary.main', fontSize: 20 }} />
              <Typography variant="h5" sx={{ fontWeight: 700 }}>
                Options Chain
              </Typography>
              <Chip label="AAPL" size="small" sx={{ height: 20, fontSize: '0.65rem', fontWeight: 700, bgcolor: alpha('#10b981', 0.1), color: 'success.main' }} />
            </Stack>
            <Suspense fallback={<LoadingFallback />}>
              <OptionsChain symbol="AAPL" />
            </Suspense>
          </Paper>
        </Grid>

        {/* Greeks + Vol surface */}
        <Grid size={{ xs: 12, lg: 4 }}>
          <Stack spacing={3}>
            <Paper data-testid="greeks-heatmap-paper" sx={{ p: 3, height: 285 }}>
              <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
                <GreeksIcon sx={{ color: theme.palette.financial?.greeks?.delta ?? '#38bdf8', fontSize: 18 }} />
                <Typography variant="body2" sx={{ fontWeight: 700 }}>Greeks · Delta</Typography>
              </Stack>
              <Suspense fallback={<LoadingFallback />}>
                <GreeksHeatmap symbol="AAPL" greek="delta" />
              </Suspense>
            </Paper>
            <Paper data-testid="volatility-surface-paper" sx={{ p: 3, height: 285 }}>
              <Typography variant="body2" sx={{ fontWeight: 700, mb: 2 }}>
                Volatility Surface
              </Typography>
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
