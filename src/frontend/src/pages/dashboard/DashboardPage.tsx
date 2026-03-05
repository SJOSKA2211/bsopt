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
  Zap,
  Globe,
  Layers,
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
// KPI Card – QFD Enhanced
// ---------------------------------------------------------------------------
interface KpiCardProps {
  label: string;
  value: string;
  subValue?: string;
  positive?: boolean;
  neutral?: boolean;
  icon: React.ReactNode;
  accentColor: string;
  progress?: number;
  greek?: string;
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
  greek = 'Δ',
}) => {
  const theme = useTheme();
  const qfd = theme.palette.financial?.qfd;

  return (
    <Paper
      elevation={0}
      sx={{
        p: 3,
        position: 'relative',
        overflow: 'hidden',
        height: '100%',
        background: alpha(theme.palette.background.paper, 0.4),
        backdropFilter: 'blur(30px) saturate(180%)',
        border: `0.5px solid ${alpha(accentColor, 0.2)}`,
        '&:hover': {
          borderColor: accentColor,
          boxShadow: `0 0 30px ${alpha(accentColor, 0.15)}`,
          '& .card-icon': {
            transform: 'scale(1.1) rotate(5deg)',
            color: accentColor,
          },
          '& .greek-overlay': {
            opacity: 0.15,
            transform: 'scale(1.1)',
          }
        }
      }}
    >
      {/* Iridescent Border Line */}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: 2,
          background: `linear-gradient(90deg, transparent, ${accentColor}, transparent)`,
          opacity: 0.8
        }}
      />

      {/* Background Greek Character */}
      <Typography
        className="greek-overlay"
        sx={{
          position: 'absolute',
          bottom: -20,
          right: -10,
          fontSize: '120px',
          fontWeight: 900,
          color: accentColor,
          opacity: 0.05,
          fontFamily: 'Outfit, sans-serif',
          zIndex: 0,
          transition: 'all 0.4s ease',
          pointerEvents: 'none',
        }}
      >
        {greek}
      </Typography>

      <Stack spacing={2} sx={{ position: 'relative', zIndex: 1 }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Typography
            variant="caption"
            sx={{
              color: 'text.secondary',
              fontWeight: 800,
              letterSpacing: '0.15em',
              fontFamily: 'Outfit, sans-serif'
            }}
          >
            {label}
          </Typography>
          <Box
            className="card-icon"
            sx={{
              color: alpha(accentColor, 0.7),
              transition: 'all 0.3s ease',
              display: 'flex'
            }}
          >
            {icon}
          </Box>
        </Stack>

        <Box>
          <Typography
            variant="h3"
            sx={{
              fontWeight: 800,
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '1.75rem',
              color: theme.palette.text.primary,
              letterSpacing: '-0.02em'
            }}
          >
            {value}
          </Typography>

          <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: 1 }}>
            {subValue && (
              <Chip
                size="small"
                label={subValue}
                variant="filled"
                sx={{
                  height: 20,
                  fontSize: '0.65rem',
                  fontWeight: 800,
                  bgcolor: alpha(positive ? theme.palette.success.main : neutral ? theme.palette.text.disabled : theme.palette.error.main, 0.1),
                  color: positive ? theme.palette.success.main : neutral ? theme.palette.text.secondary : theme.palette.error.main,
                  border: 'none',
                  fontFamily: 'JetBrains Mono'
                }}
              />
            )}
            {positive !== undefined && (
              <Typography variant="caption" sx={{ color: positive ? 'success.main' : 'error.main', fontWeight: 700 }}>
                {positive ? '↑' : '↓'}
              </Typography>
            )}
          </Stack>
        </Box>

        {progress !== undefined && (
          <Box>
            <LinearProgress
              variant="determinate"
              value={progress}
              sx={{
                height: 4,
                borderRadius: 2,
                bgcolor: alpha(accentColor, 0.1),
                '& .MuiLinearProgress-bar': {
                  borderRadius: 2,
                  background: `linear-gradient(90deg, ${accentColor}, ${alpha(accentColor, 0.5)})`,
                },
              }}
            />
          </Box>
        )}
      </Stack>
    </Paper>
  );
};

// ---------------------------------------------------------------------------
// Dashboard Page – Quantum Financial Deity Evolution
// ---------------------------------------------------------------------------
export const DashboardPage: React.FC = () => {
  const theme = useTheme();
  const [activeTime, setActiveTime] = React.useState('1M');
  const qfd = theme.palette.financial?.qfd;

  return (
    <Box sx={{ maxWidth: 1600, mx: 'auto', px: { xs: 2, md: 4 }, pb: 8 }}>
      {/* ---- Divine Header ---- */}
      <Stack
        direction={{ xs: 'column', sm: 'row' }}
        justifyContent="space-between"
        alignItems={{ xs: 'flex-start', sm: 'center' }}
        sx={{ mb: 6, mt: 2 }}
        spacing={2}
      >
        <Box>
          <Typography
            variant="h2"
            sx={{
              fontWeight: 900,
              mb: 0.5,
              fontFamily: 'Outfit, sans-serif',
              letterSpacing: '-0.04em'
            }}
          >
            Welcome,{' '}
            <Box
              component="span"
              sx={{
                background: `linear-gradient(135deg, ${qfd?.quantum}, ${qfd?.nebula})`,
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              Arch-Quant
            </Box>
          </Typography>
          <Stack direction="row" spacing={2} alignItems="center">
            <Typography variant="body2" sx={{ color: 'text.secondary', fontWeight: 500 }}>
              The Quantum Field is stable. All systems operational.
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Chip label="mTLS Secured" size="small" icon={<Globe sx={{ fontSize: '12px !important' }} />} sx={{ height: 20, fontSize: '10px', fontWeight: 700 }} />
              <Chip label="Real-time BS-WASM" size="small" icon={<Layers sx={{ fontSize: '12px !important' }} />} sx={{ height: 20, fontSize: '10px', fontWeight: 700 }} />
            </Box>
          </Stack>
        </Box>
        <Stack direction="row" spacing={2}>
          <Button
            variant="outlined"
            startIcon={<GreeksIcon />}
            sx={{
              borderColor: alpha(qfd?.nebula ?? '#7B68EE', 0.5),
              color: qfd?.nebula,
              '&:hover': { borderColor: qfd?.nebula, bgcolor: alpha(qfd?.nebula ?? '#7B68EE', 0.05) }
            }}
          >
            Analytics
          </Button>
          <Button
            variant="contained"
            startIcon={<Zap />}
            color="primary"
            sx={{ boxShadow: `0 0 20px ${alpha(qfd?.quantum ?? '#00FFFF', 0.3)}` }}
          >
            Execute Trade
          </Button>
        </Stack>
      </Stack>

      {/* ---- Quantum KPI Grid ---- */}
      <Grid container spacing={3} sx={{ mb: 6 }}>
        <Grid item xs={12} md={4}>
          <KpiCard
            label="Total Value"
            value="$1,248,392.42"
            subValue="+$42,109 (3.4%)"
            positive
            icon={<PortfolioIcon />}
            accentColor={qfd?.quantum ?? '#00FFFF'}
            greek="Σ"
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <KpiCard
            label="Active Gamma"
            value="342.18"
            subValue="-12.4 today"
            icon={<GreeksIcon />}
            accentColor={qfd?.nebula ?? '#7B68EE'}
            greek="Γ"
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <KpiCard
            label="ML Confidence"
            value="98.2%"
            subValue="Calibration Optimal"
            icon={<MLIcon />}
            accentColor={qfd?.electrum ?? '#D4AF37'}
            progress={98}
            greek="Ψ"
          />
        </Grid>
      </Grid>

      {/* ---- Trading Cockpit ---- */}
      <Grid container spacing={3}>
        {/* Main Chart Section */}
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 4, height: 600, display: 'flex', flexDirection: 'column' }}>
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 4 }}>
              <Box>
                <Typography variant="h4" sx={{ fontWeight: 800 }}>Market Trajectory</Typography>
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>Live feed from direct exchange bridge</Typography>
              </Box>
              <Stack direction="row" spacing={1} sx={{ bgcolor: alpha('#f8fafc', 0.05), p: 0.5, borderRadius: 2 }}>
                {['1H', '4H', '1D', '1W', 'ALL'].map((t) => (
                  <Button
                    key={t}
                    size="small"
                    variant={t === activeTime ? 'contained' : 'text'}
                    onClick={() => setActiveTime(t)}
                    sx={{
                      minWidth: 48,
                      height: 32,
                      fontSize: '0.7rem',
                      fontWeight: 700,
                      ...(t !== activeTime && { color: 'text.secondary' })
                    }}
                  >
                    {t}
                  </Button>
                ))}
              </Stack>
            </Stack>
            <Box sx={{ flexGrow: 1, minHeight: 400 }}>
              <Suspense fallback={<LoadingFallback />}>
                <LivePriceChart symbol="SPX" />
              </Suspense>
            </Box>
          </Paper>
        </Grid>

        {/* Intelligence Sidebar */}
        <Grid item xs={12} lg={4}>
          <Stack spacing={3}>
            <Paper sx={{ p: 3, height: 285 }}>
              <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
                <MLIcon sx={{ color: qfd?.electrum, fontSize: 20 }} />
                <Typography variant="h6" sx={{ fontWeight: 800 }}>Neural Inference</Typography>
              </Stack>
              <Suspense fallback={<LoadingFallback />}>
                <MLPredictions symbol="SPX" />
              </Suspense>
            </Paper>
            <Paper sx={{ p: 3, height: 285 }}>
              <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
                <GreeksIcon sx={{ color: qfd?.quantum, fontSize: 20 }} />
                <Typography variant="h6" sx={{ fontWeight: 800 }}>Greeks Surface</Typography>
              </Stack>
              <Suspense fallback={<LoadingFallback />}>
                <GreeksHeatmap symbol="SPX" greek="delta" />
              </Suspense>
            </Paper>
          </Stack>
        </Grid>

        {/* Options Engine */}
        <Grid item xs={12}>
          <Paper sx={{ height: 600, overflow: 'hidden' }}>
            <Stack direction="row" spacing={2} alignItems="center" sx={{ p: 4, pb: 2 }}>
              <Layers sx={{ color: theme.palette.primary.main }} />
              <Box>
                <Typography variant="h4" sx={{ fontWeight: 800 }}>Strategic Options Matrix</Typography>
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>Real-time WASM BS calculation with SIMD acceleration</Typography>
              </Box>
            </Stack>
            <Suspense fallback={<LoadingFallback />}>
              <OptionsChain symbol="SPX" />
            </Suspense>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DashboardPage;
