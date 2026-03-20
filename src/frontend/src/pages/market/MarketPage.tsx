import React, { lazy, Suspense, useState, useMemo, useCallback } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
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
import { usePricingStore } from '../../store/usePricingStore';
import { useDataIntegration } from '../../hooks/useDataIntegration';

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



export const MarketPage: React.FC = () => {
  const theme = useTheme();
  const [symbol, setSymbol] = useState(0);

  const handleSymbolChange = useCallback((_: React.SyntheticEvent, newValue: number) => {
    setSymbol(newValue);
  }, []);

  const currentSymbol = SYMBOLS[symbol];
  
  // Establish unified real-time connection
  const { isConnected } = useDataIntegration({ symbols: [currentSymbol] });
  
  // Get live price and stats from high perf store
  const priceData = usePricingStore((state: any) => state.prices[currentSymbol]);
  const livePrice = priceData?.price ?? 0;

  // Compute reference price from real prev_close data
  const prevClose = priceData?.prev_close ?? priceData?.price ?? 0;
  const isPriceUp = prevClose > 0 ? livePrice >= prevClose : true;
  const priceDiff = prevClose > 0 ? Math.abs(livePrice - prevClose).toFixed(2) : '0.00';
  const pricePct = prevClose > 0 ? (Math.abs(livePrice - prevClose) / prevClose * 100).toFixed(2) : '0.00';

  // Compute market pulse stats from live data
  const ivRank = priceData?.iv_rank != null ? `${(priceData.iv_rank * 100).toFixed(1)}%` : '—';
  const hv30 = priceData?.hv30 != null ? `${(priceData.hv30 * 100).toFixed(1)}%` : '—';
  const pcr = priceData?.put_call_ratio != null ? priceData.put_call_ratio.toFixed(2) : '—';
  const vol = priceData?.volume != null ? `${(priceData.volume / 1_000_000).toFixed(1)}M` : '—';
  const oi = priceData?.open_interest != null ? `${(priceData.open_interest / 1_000_000_000).toFixed(2)}B` : '—';

  const MARKET_PULSE = useMemo(() => [
    { label: 'IV Rank', value: ivRank, type: 'nebula', icon: <IVIcon sx={{ fontSize: 14 }} /> },
    { label: 'HV30', value: hv30, type: 'sky', icon: <HVIcon sx={{ fontSize: 14 }} /> },
    { label: 'P/C Ratio', value: pcr, type: 'quantum', icon: <PCRIcon sx={{ fontSize: 14 }} /> },
    { label: 'Volume', value: vol, type: 'electrum', icon: null },
    { label: 'Open Int.', value: oi, type: 'electrum', icon: null },
  ], [ivRank, hv30, pcr, vol, oi]);

  return (
    <Container maxWidth="xl" sx={{ mt: 2, pb: 6 }}>
      {/* Header */}
      <Stack direction="row" justifyContent="space-between" alignItems="flex-end" sx={{ mb: 3 }}>
        <Box>
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
          >
            <Typography
              variant="h3"
              sx={{
                fontWeight: 900,
                mb: 0.5,
                fontFamily: 'Outfit',
                letterSpacing: '-0.04em',
                background: theme.palette.financial.qfd.iridescent,
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                textShadow: `0 0 40px ${alpha(theme.palette.financial.qfd.quantum, 0.3)}`,
              }}
            >
              Market Data
            </Typography>
          </motion.div>
          <Typography variant="body2" sx={{ color: 'text.secondary', fontWeight: 500 }}>
            Real-time options analytics & volatility surface
          </Typography>
        </Box>
        <Chip
          icon={<Box component="span" className={isConnected ? "live-dot" : ""} sx={{ mr: 1, height: 8, width: 8, borderRadius: '50%', bgcolor: isConnected ? theme.palette.success.main : theme.palette.warning.main, filter: isConnected ? `drop-shadow(0 0 5px ${theme.palette.success.main})` : 'none' }} />}
          label={isConnected ? "STREAMING" : "CONNECTING..."}
          size="small"
          sx={{
            bgcolor: alpha(isConnected ? theme.palette.success.main : theme.palette.warning.main, 0.1),
            color: isConnected ? theme.palette.success.main : theme.palette.warning.main,
            border: `1px solid ${alpha(isConnected ? theme.palette.success.main : theme.palette.warning.main, 0.2)}`,
            fontWeight: 700,
            fontSize: '0.65rem',
            letterSpacing: '0.07em',
            '& .MuiChip-icon': { ml: 1, mr: -0.5, width: 'auto', height: 'auto' },
          }}
        />
      </Stack>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.1 }}
      >
        <Tabs
          value={symbol}
          onChange={handleSymbolChange}
          sx={{
            mb: 3,
            '& .MuiTab-root': {
              minWidth: 80,
              fontWeight: 800,
              fontSize: '0.85rem',
              letterSpacing: '0.05em',
              color: 'text.secondary',
              textTransform: 'none',
              transition: 'all 0.2s',
              '&:hover': { color: 'primary.main', opacity: 1 },
            },
            '& .Mui-selected': { color: 'primary.main !important' },
            '& .MuiTabs-indicator': {
              background: `linear-gradient(90deg, ${theme.palette.financial.qfd.quantum}, ${theme.palette.financial.qfd.nebula})`,
              borderRadius: 3,
              height: 4,
              boxShadow: `0 0 15px ${alpha(theme.palette.financial.qfd.quantum, 0.5)}`,
            },
          }}
        >
          {SYMBOLS.map(s => (
            <Tab key={s} label={s} />
          ))}
        </Tabs>
      </motion.div>

      {/* Market Pulse strip */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <Paper
          sx={{
            p: 0,
            mb: 4,
            borderRadius: 6,
            overflow: 'hidden',
            background: `linear-gradient(135deg, ${alpha('#0f172a', 0.6)}, ${alpha('#0f172a', 0.2)})`,
            backdropFilter: 'blur(40px) saturate(200%)',
            border: `1px solid ${alpha('#f8fafc', 0.08)}`,
          }}
        >
          <Stack
            direction="row"
            divider={
              <Box sx={{ width: 1, bgcolor: alpha('#94a3b8', 0.1) }} />
            }
          >
            {MARKET_PULSE.map((stat: any) => {
              const qfd = theme.palette.financial.qfd as Record<string, string>;
              const accentColor = qfd[stat.type] || theme.palette.text.primary;
              return (
                <Box
                  key={stat.label}
                  sx={{
                    flex: 1,
                    py: 2.5,
                    px: 2.5,
                    textAlign: 'center',
                    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                    '&:hover': {
                      bgcolor: alpha(accentColor, 0.05),
                      transform: 'scale(1.02)',
                    },
                  }}
                >
                  <Stack direction="row" spacing={1} justifyContent="center" alignItems="center" sx={{ mb: 1 }}>
                    {stat.icon && (
                      <Box sx={{ color: accentColor, display: 'flex', filter: `drop-shadow(0 0 5px ${accentColor})` }}>{stat.icon}</Box>
                    )}
                    <Typography
                      variant="caption"
                      sx={{ color: 'text.secondary', fontWeight: 800, letterSpacing: '0.12em', fontSize: '0.65rem', textTransform: 'uppercase' }}
                    >
                      {stat.label}
                    </Typography>
                  </Stack>
                  <Typography
                    variant="body2"
                    sx={{
                      fontWeight: 900,
                      color: accentColor,
                      fontFamily: 'JetBrains Mono',
                      fontSize: '1.15rem',
                      letterSpacing: '-0.02em'
                    }}
                  >
                    {stat.value}
                  </Typography>
                </Box>
              );
            })}
          </Stack>
        </Paper>
      </motion.div>

      {/* Charts */}
      <Grid container spacing={3}>
        <Grid size={{ xs: 12 }} className="slide-up" style={{ animationDelay: '0.1s' }}>
          <Paper
            sx={{
              p: 3,
              height: 520,
              borderRadius: 6,
              background: `linear-gradient(135deg, ${alpha('#0f172a', 0.6)}, ${alpha('#0f172a', 0.2)})`,
              backdropFilter: 'blur(40px) saturate(200%)',
              border: `1px solid ${alpha('#f8fafc', 0.08)}`,
            }}
          >
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
              <Stack direction="row" spacing={2} alignItems="baseline">
                <Typography variant="h5" sx={{ fontWeight: 900, fontFamily: 'Outfit' }}>
                  {currentSymbol}
                </Typography>
                <Typography
                  sx={{ fontFamily: 'JetBrains Mono', fontWeight: 800, fontSize: '1.4rem', color: isPriceUp ? 'success.main' : 'error.main' }}
                >
                  ${livePrice.toFixed(2)}
                </Typography>
                <Chip
                  label={`${isPriceUp ? '▲' : '▼'} $${priceDiff} (${pricePct}%)`}
                  size="small"
                  sx={{
                    height: 24,
                    fontSize: '0.75rem',
                    fontWeight: 800,
                    bgcolor: alpha(isPriceUp ? theme.palette.success.main : theme.palette.error.main, 0.1),
                    color: isPriceUp ? 'success.main' : 'error.main',
                    border: `1px solid ${alpha(isPriceUp ? theme.palette.success.main : theme.palette.error.main, 0.2)}`,
                    borderRadius: 1.5
                  }}
                />
              </Stack>
            </Stack>
            <Box sx={{ height: 440, borderRadius: 4, overflow: 'hidden', border: `1px solid ${alpha('#fff', 0.03)}` }}>
              <Suspense fallback={<LoadingFallback />}>
                <LivePriceChart symbol={currentSymbol} />
              </Suspense>
            </Box>
          </Paper>
        </Grid>

        <Grid size={{ xs: 12, lg: 8 }} className="slide-up" style={{ animationDelay: '0.2s' }}>
          <Paper
            sx={{
              height: 600,
              overflow: 'hidden',
              borderRadius: 6,
              background: `linear-gradient(135deg, ${alpha('#0f172a', 0.6)}, ${alpha('#0f172a', 0.2)})`,
              backdropFilter: 'blur(40px) saturate(200%)',
              border: `1px solid ${alpha('#f8fafc', 0.08)}`,
            }}
          >
            <Stack direction="row" spacing={1.5} alignItems="center" sx={{ p: 3, pb: 2 }}>
              <Typography variant="h5" sx={{ fontWeight: 900, fontFamily: 'Outfit' }}>
                Options Chain
              </Typography>
              <Chip
                label={currentSymbol}
                size="small"
                sx={{
                  height: 22,
                  fontSize: '0.7rem',
                  fontWeight: 800,
                  bgcolor: alpha(theme.palette.primary.main, 0.1),
                  color: 'primary.main',
                  border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
                  borderRadius: 1.5,
                }}
              />
            </Stack>
            <Suspense fallback={<LoadingFallback />}>
              <OptionsChain symbol={currentSymbol} />
            </Suspense>
          </Paper>
        </Grid>

        <Grid size={{ xs: 12, lg: 4 }} className="slide-up" style={{ animationDelay: '0.3s' }}>
          <Paper
            sx={{
              p: 3,
              height: 600,
              borderRadius: 6,
              background: `linear-gradient(135deg, ${alpha('#0f172a', 0.6)}, ${alpha('#0f172a', 0.2)})`,
              backdropFilter: 'blur(40px) saturate(200%)',
              border: `1px solid ${alpha('#f8fafc', 0.08)}`,
            }}
          >
            <Typography variant="h5" sx={{ fontWeight: 900, mb: 3, fontFamily: 'Outfit' }}>
              Greeks · Delta Heatmap
            </Typography>
            <Box sx={{ borderRadius: 4, overflow: 'hidden', border: `1px solid ${alpha('#fff', 0.03)}` }}>
              <AnimatePresence mode="wait">
                <motion.div
                  key={currentSymbol}
                  initial={{ opacity: 0, scale: 0.98 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 1.02 }}
                  transition={{ duration: 0.2 }}
                >
                  <Suspense fallback={<LoadingFallback />}>
                    <GreeksHeatmap symbol={currentSymbol} greek="delta" />
                  </Suspense>
                </motion.div>
              </AnimatePresence>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default MarketPage;
