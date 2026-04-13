import React from 'react';
import {
  Box,
  Typography,
  Stack,
  Divider,
  CircularProgress,
  useTheme,
  alpha,
  Paper,
  Chip,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  AccountBalanceWallet,
  ShieldMoon,
  Timeline,
} from '@mui/icons-material';
import { usePortfolio } from '../hooks/usePortfolio';

export const PortfolioSummary: React.FC = React.memo(() => {
  const theme = useTheme();
  // Midnight Emerald Theme Access
  const financial = (theme.palette as any).financial;
  const qfd = financial?.qfd;
  
  const { data, isLoading, isError } = usePortfolio();

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', minHeight: 250 }}>
        <CircularProgress size={32} thickness={5} sx={{ color: qfd?.emerald }} aria-label="Synchronizing Portfolio..." />
      </Box>
    );
  }

  if (isError || !data) {
    return (
      <Paper sx={{ p: 4, textAlign: 'center', bgcolor: alpha(theme.palette.error.main, 0.05), borderRadius: 6, border: `1px solid ${alpha(theme.palette.error.main, 0.1)}` }}>
        <Typography color="error" variant="body2" sx={{ fontWeight: 800 }}>PORTFOLIO MANIFOLD DISCONNECTED</Typography>
      </Paper>
    );
  }

  const { totalValue, dailyPnL, dailyPnLPercent, positionsCount } = data;
  const isPositive = dailyPnL >= 0;

  return (
    <Paper
      className="qfd-glass"
      sx={{
        p: 3,
        borderRadius: 8,
        bgcolor: alpha(theme.palette.background.paper, 0.1),
        border: `1px solid ${alpha('#fff', 0.05)}`,
        backdropFilter: 'blur(40px)',
        position: 'relative',
        overflow: 'hidden',
        boxShadow: `0 20px 40px ${alpha('#000', 0.4)}`,
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 4 }}>
        <Stack direction="row" spacing={1.5} alignItems="center">
          <Box sx={{ p: 1, borderRadius: 2, bgcolor: alpha(qfd?.emerald ?? '#10b981', 0.1) }}>
            <AccountBalanceWallet sx={{ color: qfd?.emerald, fontSize: 24 }} />
          </Box>
          <Box>
            <Typography variant="subtitle2" sx={{ fontWeight: 900, fontFamily: 'Outfit', letterSpacing: '-0.01em' }}>
              Portfolio Summary
            </Typography>
            <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 700, letterSpacing: '0.05em' }}>QUANT MANIFOLD</Typography>
          </Box>
        </Stack>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ShieldMoon sx={{ fontSize: 16, color: qfd?.amber }} />
          <Typography variant="caption" sx={{ fontWeight: 800, color: qfd?.amber }}>SECURE</Typography>
        </Box>
      </Stack>

      <Stack spacing={4} sx={{ flexGrow: 1 }}>
        <Box>
          <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 800, letterSpacing: '0.1em', mb: 1, display: 'block' }}>
            TOTAL LIQUIDITY
          </Typography>
          <Typography variant="h3" sx={{ 
            fontWeight: 900, 
            fontFamily: 'JetBrains Mono', 
            color: 'text.primary',
            letterSpacing: '-0.05em',
            textShadow: `0 0 30px ${alpha(qfd?.emerald ?? '#10b981', 0.2)}`
          }}>
            ${totalValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </Typography>
        </Box>

        <Divider sx={{ borderStyle: 'solid', opacity: 0.1 }} />

        <Stack direction="row" spacing={4}>
          <Box sx={{ flex: 1 }}>
            <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 800, letterSpacing: '0.1em', mb: 1, display: 'block' }}>
              DAILY PERFORMANCE
            </Typography>
            <Stack direction="row" spacing={1} alignItems="center">
              {isPositive ? (
                <TrendingUp sx={{ color: qfd?.emerald, fontSize: 20 }} />
              ) : (
                <TrendingDown sx={{ color: theme.palette.error.main, fontSize: 20 }} />
              )}
              <Typography
                variant="h5"
                sx={{ 
                  fontWeight: 900, 
                  fontFamily: 'JetBrains Mono',
                  color: isPositive ? qfd?.emerald : theme.palette.error.main,
                  letterSpacing: '-0.03em'
                }}
              >
                {isPositive ? '+' : ''}${Math.abs(dailyPnL).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </Typography>
            </Stack>
          </Box>

          <Box sx={{ flex: 1 }}>
            <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 800, letterSpacing: '0.1em', mb: 1, display: 'block' }}>
              ALPHA BASIS
            </Typography>
            <Typography
              variant="h5"
              sx={{ 
                fontWeight: 900, 
                fontFamily: 'JetBrains Mono',
                color: isPositive ? qfd?.emerald : theme.palette.error.main,
                letterSpacing: '-0.03em'
              }}
            >
              {isPositive ? '+' : ''}{dailyPnLPercent.toFixed(2)}%
            </Typography>
          </Box>
        </Stack>

        <Box sx={{ 
          mt: 'auto', 
          p: 2, 
          borderRadius: 4, 
          bgcolor: alpha('#fff', 0.02),
          border: `1px solid ${alpha('#fff', 0.03)}`
        }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Stack direction="row" spacing={1.5} alignItems="center">
              <Timeline sx={{ fontSize: 18, color: qfd?.emerald }} />
              <Box>
                <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 800, display: 'block' }}>
                  ACTIVE POSITIONS
                </Typography>
                <Typography variant="body2" sx={{ fontWeight: 900, fontFamily: 'JetBrains Mono' }}>
                  {positionsCount} UNITS
                </Typography>
              </Box>
            </Stack>
            <Chip 
              label="SYNCHRONIZED" 
              size="small" 
              sx={{ 
                height: 20, 
                fontSize: '0.6rem', 
                fontWeight: 900, 
                bgcolor: alpha(qfd?.emerald ?? '#10b981', 0.1),
                color: qfd?.emerald,
                border: `1px solid ${alpha(qfd?.emerald ?? '#10b981', 0.2)}`
              }} 
            />
          </Stack>
        </Box>
      </Stack>
    </Paper>
  );
});
