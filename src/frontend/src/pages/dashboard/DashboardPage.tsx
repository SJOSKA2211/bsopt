import React, { lazy, Suspense } from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  Stack,
  alpha,
  useTheme,
  Chip,
  Button,
  CircularProgress,
  LinearProgress,
} from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
import { useMotion } from '../../hooks/useMotion';
import {
  GreeksIcon,
  MLIcon,
  PortfolioIcon,
  Zap,
  Globe,
  Layers,
  TrendingUp,
} from '../../components/common/Icons'; // Assuming these exist or using Mui icons
import {
  TrendingUp as TrendingUpIcon,
  WaterfallChart as GreeksIconMui,
  Bolt as MLIconMui,
  AccountBalance as PortfolioIconMui,
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

const LoadingFallback = () => (
  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', p: 4 }}>
    <CircularProgress size={28} />
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
  const { variants } = useMotion();

  return (
    <motion.div
      variants={variants.glassCard}
      whileHover="hover"
      style={{ height: '100%' }}
    >
      <Paper
        className="qfd-glass"
        elevation={0}
        sx={{
          p: 3,
          position: 'relative',
          overflow: 'hidden',
          height: '100%',
          borderRadius: 6,
          border: `1px solid ${alpha(accentColor, 0.1)}`,
          background: `linear-gradient(135deg, ${alpha('#0f172a', 0.6)}, ${alpha('#0f172a', 0.2)})`,
          backdropFilter: 'blur(40px) saturate(200%)',
          transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            borderColor: alpha(accentColor, 0.5),
            boxShadow: `0 20px 40px -10px ${alpha(accentColor, 0.2)}`,
            '& .card-icon': {
              transform: 'scale(1.2) rotate(15deg)',
              color: accentColor,
              filter: `drop-shadow(0 0 10px ${accentColor})`,
            },
            '& .greek-overlay': {
              opacity: 0.15,
              transform: 'scale(1.3) rotate(-10deg)',
            }
          }
        }}
      >
        {/* Iridescent Top Beam */}
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: 3,
            background: `linear-gradient(90deg, transparent, ${accentColor}, transparent)`,
            filter: 'blur(1px)',
          }}
        />

        {/* Large Greek Glyph background */}
        <Typography
          className="greek-overlay"
          sx={{
            position: 'absolute',
            bottom: -30,
            right: -20,
            fontSize: '140px',
            fontWeight: 900,
            color: accentColor,
            opacity: 0.03,
            transition: 'all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)',
            pointerEvents: 'none',
            zIndex: 0,
            userSelect: 'none',
          }}
        >
          {greek}
        </Typography>

        <Stack spacing={2.5} sx={{ position: 'relative', zIndex: 1 }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Typography
              variant="overline"
              sx={{
                color: alpha(theme.palette.text.primary, 0.5),
                fontWeight: 900,
                letterSpacing: '0.2em',
                fontSize: '0.65rem'
              }}
            >
              {label}
            </Typography>
            <Box
              className="card-icon"
              sx={{
                color: alpha(accentColor, 0.6),
                transition: 'all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)',
                display: 'flex',
                fontSize: 24
              }}
            >
              {icon}
            </Box>
          </Stack>

          <Box>
            <Typography
              variant="h3"
              sx={{
                fontWeight: 900,
                fontFamily: 'JetBrains Mono',
                fontSize: '1.85rem',
                letterSpacing: '-0.02em',
                mb: 0.5
              }}
            >
              {value}
            </Typography>

            <Stack direction="row" spacing={1} alignItems="center">
              {subValue && (
                <Box
                  sx={{
                    px: 1,
                    py: 0.25,
                    borderRadius: 1,
                    bgcolor: alpha(positive ? theme.palette.success.main : neutral ? theme.palette.text.disabled : theme.palette.error.main, 0.1),
                    border: `1px solid ${alpha(positive ? theme.palette.success.main : neutral ? theme.palette.text.disabled : theme.palette.error.main, 0.2)}`,
                  }}
                >
                  <Typography
                    sx={{
                      fontSize: '0.7rem',
                      fontWeight: 900,
                      color: positive ? theme.palette.success.main : neutral ? theme.palette.text.secondary : theme.palette.error.main,
                      fontFamily: 'JetBrains Mono'
                    }}
                  >
                    {subValue}
                  </Typography>
                </Box>
              )}
            </Stack>
          </Box>

          {progress !== undefined && (
            <Box>
              <LinearProgress
                variant="determinate"
                value={progress}
                sx={{
                  height: 6,
                  borderRadius: 3,
                  bgcolor: alpha(accentColor, 0.05),
                  '& .MuiLinearProgress-bar': {
                    borderRadius: 3,
                    background: `linear-gradient(90deg, ${accentColor}, ${alpha(accentColor, 0.5)})`,
                    boxShadow: `0 0 10px ${alpha(accentColor, 0.5)}`
                  },
                }}
              />
            </Box>
          )}
        </Stack>
      </Paper>
    </motion.div>
  );
};


// ---------------------------------------------------------------------------
// Dashboard Page – Quantum Financial Deity Evolution
// ---------------------------------------------------------------------------
export const DashboardPage: React.FC = () => {
  const theme = useTheme();
  const { variants } = useMotion();
  const [activeTime, setActiveTime] = React.useState('1M');
  const qfd = theme.palette.financial.qfd;

  return (
    <Box sx={{ maxWidth: 1600, mx: 'auto', px: { xs: 2, md: 4 }, pb: 8, pt: 2 }}>
      {/* ---- Divine Header ---- */}
      <motion.div variants={variants.slideUp} initial="initial" animate="animate">
        <Stack
          direction={{ xs: 'column', lg: 'row' }}
          justifyContent="space-between"
          alignItems={{ xs: 'flex-start', lg: 'center' }}
          className="qfd-glass qfd-holographic"
          sx={{ mb: 8, p: 4, borderRadius: 6, position: 'relative' }}
          spacing={4}
        >
          <Box>
            <Typography
              variant="h1"
              sx={{
                fontWeight: 950,
                fontSize: { xs: '2.5rem', md: '3.5rem' },
                mb: 1,
                fontFamily: 'Outfit',
                letterSpacing: '-0.05em',
                lineHeight: 1
              }}
            >
              Salutations,{' '}
              <Box
                component="span"
                sx={{
                  background: `linear-gradient(135deg, ${qfd?.quantum ?? '#00FFFF'}, ${qfd?.nebula ?? '#7B68EE'} 50%, ${qfd?.electrum ?? '#D4AF37'})`,
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundSize: '200% auto',
                  animation: 'shimmer 4s linear infinite',
                  '@keyframes shimmer': {
                    '0%': { backgroundPosition: '0% center' },
                    '100%': { backgroundPosition: '200% center' },
                  }
                }}
              >
                Arch-Quant
              </Box>
            </Typography>
            <Stack direction="row" spacing={2} alignItems="center" sx={{ opacity: 0.8 }}>
              <Typography variant="body1" sx={{ fontWeight: 600, color: 'text.secondary' }}>
                Quantum synchronization complete.
              </Typography>
              <Stack direction="row" spacing={1}>
                <Chip
                  label="P-8 mTLS"
                  size="small"
                  variant="outlined"
                  sx={{ borderRadius: 1, fontSize: '10px', fontWeight: 900, borderColor: alpha(qfd?.quantum ?? '#00FFFF', 0.3), color: qfd?.quantum }}
                />
                <Chip
                  label="SHM-LOW-LAT"
                  size="small"
                  variant="outlined"
                  sx={{ borderRadius: 1, fontSize: '10px', fontWeight: 900, borderColor: alpha(qfd?.nebula ?? '#7B68EE', 0.3), color: qfd?.nebula }}
                />
              </Stack>
            </Stack>
          </Box>
          <Stack direction="row" spacing={2}>
            <Button
              variant="text"
              startIcon={<Globe sx={{ fontSize: 18 }} />}
              sx={{
                fontWeight: 800,
                color: 'text.secondary',
                px: 3,
                '&:hover': { color: 'text.primary', bgcolor: alpha('#fff', 0.05) }
              }}
            >
              Network
            </Button>
            <Button
              variant="contained"
              disableElevation
              startIcon={<Zap />}
              sx={{
                fontWeight: 900,
                px: 4,
                py: 1.5,
                borderRadius: 3,
                bgcolor: qfd?.quantum ?? '#00FFFF',
                color: '#000',
                '&:hover': {
                  bgcolor: alpha(qfd?.quantum ?? '#00FFFF', 0.8),
                  boxShadow: `0 0 30px ${alpha(qfd?.quantum ?? '#00FFFF', 0.4)}`
                }
              }}
            >
              INITIATE EXECUTION
            </Button>
          </Stack>
        </Stack>
      </motion.div>

      {/* ---- Quantum KPI Grid ---- */}
      <motion.div variants={variants.staggerContainer} initial="initial" animate="animate">
        <Grid container spacing={3} sx={{ mb: 8 }}>
          <Grid item xs={12} md={4}>
            <KpiCard
              label="Portfolio Oracle"
              value="$1,248,392.42"
              subValue="+$42,109 (3.4%)"
              positive
              icon={<PortfolioIconMui />}
              accentColor={qfd?.quantum ?? '#00FFFF'}
              greek="Π"
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <KpiCard
              label="Systemic Gamma"
              value="342.18"
              subValue="-12.4 today"
              icon={<GreeksIconMui />}
              accentColor={qfd?.nebula ?? '#7B68EE'}
              greek="Γ"
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <KpiCard
              label="Predictive Accuracy"
              value="98.2%"
              subValue="Model: Heston-XL"
              icon={<MLIconMui />}
              accentColor={qfd?.electrum ?? '#D4AF37'}
              progress={98}
              greek="Φ"
            />
          </Grid>
        </Grid>
      </motion.div>


      {/* ---- Intelligence Cluster ---- */}
      <Grid container spacing={4}>
        {/* Main Observation Deck */}
        <Grid item xs={12} xl={8}>
          <motion.div variants={variants.slideUp} initial="initial" animate="animate">
            <Paper
              className="qfd-glass"
              sx={{
                p: 4,
                height: 650,
                display: 'flex',
                flexDirection: 'column',
                borderRadius: 6,
                border: `1px solid ${alpha('#fff', 0.05)}`
              }}
            >
              <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 4 }}>
                <Box>
                  <Typography variant="h4" sx={{ fontWeight: 900, letterSpacing: '-0.03em' }}>Temporal Trajectory</Typography>
                  <Typography variant="body2" sx={{ color: 'text.secondary', fontWeight: 600 }}>Live market telemetry from SHM Ringbuffer</Typography>
                </Box>
                <Stack direction="row" spacing={1} sx={{ bgcolor: alpha('#000', 0.2), p: 0.5, borderRadius: 3 }}>
                  {['1H', '4H', '1D', '1W', 'ALL'].map((t) => (
                    <Button
                      key={t}
                      size="small"
                      variant={t === activeTime ? 'contained' : 'text'}
                      onClick={() => setActiveTime(t)}
                      sx={{
                        minWidth: 50,
                        height: 32,
                        borderRadius: 2,
                        fontSize: '0.7rem',
                        fontWeight: 900,
                        ...(t === activeTime ? {
                          bgcolor: alpha(qfd?.quantum ?? '#00FFFF', 0.2),
                          color: qfd?.quantum,
                          border: `1px solid ${alpha(qfd?.quantum ?? '#00FFFF', 0.3)}`
                        } : {
                          color: 'text.secondary'
                        })
                      }}
                    >
                      {t}
                    </Button>
                  ))}
                </Stack>
              </Stack>
              <Box sx={{ flexGrow: 1, border: `1px solid ${alpha('#fff', 0.03)}`, borderRadius: 4, overflow: 'hidden' }}>
                <Suspense fallback={<LoadingFallback />}>
                  <LivePriceChart symbol="SPX" />
                </Suspense>
              </Box>
            </Paper>
          </motion.div>
        </Grid>


        {/* Cognitive Sidebars */}
        <Grid item xs={12} xl={4}>
          <motion.div variants={variants.staggerContainer} initial="initial" animate="animate">
            <Stack spacing={4}>
              <motion.div variants={variants.slideUp}>
                <Paper className="qfd-glass" sx={{ p: 3, height: 305, borderRadius: 5 }}>
                  <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 2.5 }}>
                    <Box sx={{ p: 1, borderRadius: 2, bgcolor: alpha(qfd?.electrum ?? '#D4AF37', 0.1) }}>
                      <MLIconMui sx={{ color: qfd?.electrum, fontSize: 20 }} />
                    </Box>
                    <Typography variant="h6" sx={{ fontWeight: 900 }}>Neural Inference</Typography>
                  </Stack>
                  <Suspense fallback={<LoadingFallback />}>
                    <MLPredictions symbol="SPX" />
                  </Suspense>
                </Paper>
              </motion.div>
              <motion.div variants={variants.slideUp}>
                <Paper className="qfd-glass" sx={{ p: 3, height: 305, borderRadius: 5 }}>
                  <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 2.5 }}>
                    <Box sx={{ p: 1, borderRadius: 2, bgcolor: alpha(qfd?.nebula ?? '#7B68EE', 0.1) }}>
                      <GreeksIconMui sx={{ color: qfd?.nebula, fontSize: 20 }} />
                    </Box>
                    <Typography variant="h6" sx={{ fontWeight: 900 }}>Greeks Surface</Typography>
                  </Stack>
                  <Suspense fallback={<LoadingFallback />}>
                    <GreeksHeatmap symbol="SPX" greek="delta" />
                  </Suspense>
                </Paper>
              </motion.div>
            </Stack>
          </motion.div>
        </Grid>


        {/* Transdimensional Matrix (Options) */}
        <Grid item xs={12}>
          <motion.div variants={variants.slideUp} initial="initial" animate="animate">
            <Paper
              className="qfd-glass"
              sx={{
                height: 700,
                overflow: 'hidden',
                borderRadius: 6,
                border: `1px solid ${alpha('#fff', 0.05)}`
              }}
            >
              <Suspense fallback={<LoadingFallback />}>
                <OptionsChain symbol="SPX" />
              </Suspense>
            </Paper>
          </motion.div>
        </Grid>

      </Grid>
    </Box>
  );
};

export default DashboardPage;
