import React from 'react';
import {
  Box,
  Typography,
  Stack,
  Chip,
  CircularProgress,
  Divider,
  alpha,
  useTheme,
  Paper,
} from '@mui/material';
import {
  Psychology,
  AutoGraph,
  Update,
  TrendingUp,
  TrendingDown,
} from '@mui/icons-material';

// Institutional API Hooks
import { useMLInference } from '../../../api/hooks';

interface MLPredictionsProps {
  symbol: string;
}

export const MLPredictions: React.FC<MLPredictionsProps> = React.memo(({ symbol }) => {
  const theme = useTheme();
  // Midnight Emerald Theme Access
  const financial = (theme.palette as any).financial;
  const qfd = financial?.qfd;

  const { data, loading: isLoading, error } = useMLInference(symbol);

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', minHeight: 250 }}>
        <CircularProgress size={32} thickness={5} sx={{ color: qfd?.emerald }} aria-label="Synchronizing Oracle..." />
      </Box>
    );
  }

  if (error || !data) {
    return (
      <Paper sx={{ p: 4, textAlign: 'center', bgcolor: alpha(theme.palette.error.main, 0.05), borderRadius: 6, border: `1px solid ${alpha(theme.palette.error.main, 0.1)}` }}>
        <Typography color="error" variant="body2" sx={{ fontWeight: 800 }}>ORACLE DISCOVERY FAILED</Typography>
      </Paper>
    );
  }

  const isPositive = (data.drift || 0) >= 0;

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
      }}
    >
      {/* Oracle Status Bar */}
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: 3,
          background: `linear-gradient(90deg, transparent, ${qfd?.amber ?? '#f59e0b'}, transparent)`,
        }}
      />

      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 3 }}>
        <Stack direction="row" spacing={1.5} alignItems="center">
          <Box sx={{ p: 1, borderRadius: 2, bgcolor: alpha(qfd?.amber ?? '#f59e0b', 0.15) }}>
            <Psychology sx={{ color: qfd?.amber, fontSize: 24 }} />
          </Box>
          <Box>
            <Typography variant="subtitle2" sx={{ fontWeight: 900, fontFamily: 'Outfit', letterSpacing: '-0.01em' }}>
              Price Oracle
            </Typography>
            <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 700, letterSpacing: '0.05em' }}>NEURAL INFERENCE</Typography>
          </Box>
        </Stack>
        <Chip
          icon={isPositive ? <TrendingUp sx={{ fontSize: '14px !important' }} /> : <TrendingDown sx={{ fontSize: '14px !important' }} />}
          label={`${isPositive ? '+' : ''}${((data.drift || 0) * 100).toFixed(2)}%`}
          size="small"
          sx={{ 
            bgcolor: alpha(isPositive ? qfd?.emerald ?? '#10b981' : theme.palette.error.main, 0.1),
            color: isPositive ? qfd?.emerald : theme.palette.error.main,
            fontWeight: 900,
            border: `1px solid ${alpha(isPositive ? qfd?.emerald ?? '#10b981' : theme.palette.error.main, 0.2)}`,
            fontFamily: 'JetBrains Mono',
            fontSize: '0.7rem'
          }}
        />
      </Stack>

      <Stack spacing={3}>
        <Box>
          <Typography variant="caption" sx={{ color: 'text.secondary', display: 'flex', alignItems: 'center', gap: 1, fontWeight: 700, mb: 1 }}>
            <AutoGraph sx={{ fontSize: 14 }} /> 24H TARGET ASYMPTOTE
          </Typography>
          <Typography variant="h3" sx={{ 
            fontWeight: 900, 
            fontFamily: 'JetBrains Mono', 
            color: qfd?.emerald,
            textShadow: `0 0 20px ${alpha(qfd?.emerald ?? '#10b981', 0.3)}`,
            letterSpacing: '-0.05em'
          }}>
            ${(data.predicted_price || 0).toLocaleString(undefined, { minimumFractionDigits: 2 })}
          </Typography>
        </Box>

        <Box sx={{ 
          bgcolor: alpha('#fff', 0.02), 
          p: 2, 
          borderRadius: 4, 
          border: `1px solid ${alpha('#fff', 0.03)}`,
          position: 'relative'
        }}>
          <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 800, letterSpacing: '0.1em', mb: 1, display: 'block' }}>
            95% CONFIDENCE MANIFOLD
          </Typography>
          <Stack direction="row" spacing={2} alignItems="center">
            <Typography variant="body2" sx={{ fontFamily: 'JetBrains Mono', fontWeight: 900 }}>
              ${(data.confidence_interval?.[0] || 0).toFixed(2)}
            </Typography>
            <Box sx={{ flex: 1, height: 2, bgcolor: alpha('#fff', 0.1), borderRadius: 1, position: 'relative' }}>
              <Box sx={{ position: 'absolute', left: '20%', right: '20%', height: '100%', bgcolor: qfd?.emerald, boxShadow: `0 0 10px ${qfd?.emerald}` }} />
            </Box>
            <Typography variant="body2" sx={{ fontFamily: 'JetBrains Mono', fontWeight: 900 }}>
              ${(data.confidence_interval?.[1] || 0).toFixed(2)}
            </Typography>
          </Stack>
        </Box>

        <Divider sx={{ opacity: 0.1 }} />

        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Box>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', fontWeight: 800, mb: 0.5 }}>
              ENGINE
            </Typography>
            <Typography variant="caption" sx={{ fontWeight: 900, color: 'text.primary', fontFamily: 'JetBrains Mono' }}>
              {data.model_name || 'NEURAL-CORE-1'}
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'right' }}>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 0.5, fontWeight: 800, mb: 0.5 }}>
              <Update sx={{ fontSize: 12 }} /> SYNCED
            </Typography>
            <Typography variant="caption" sx={{ fontFamily: 'JetBrains Mono', fontWeight: 600 }}>
              {data.last_updated ? new Date(data.last_updated).toLocaleTimeString() : 'LIVE'}
            </Typography>
          </Box>
        </Stack>
      </Stack>
    </Paper>
  );
});