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
} from '@mui/material';
import {
  Psychology,
  AutoGraph,
  Update,
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
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', minHeight: 200 }}>
        <CircularProgress size={30} aria-label="Loading predictions" />
      </Box>
    );
  }

  if (error || !data) {
    return (
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography color="error" variant="body2">ML Engine unavailable</Typography>
      </Box>
    );
  }

  const isPositive = data.drift >= 0;

  return (
    <Box sx={{ p: 2 }}>
      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
        <Psychology sx={{ color: qfd?.amber ?? 'secondary.main' }} />
        <Typography variant="subtitle1" fontWeight="bold">
          Neural Price Oracle
        </Typography>
      </Stack>

      <Stack spacing={2}>
        <Box>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <AutoGraph sx={{ fontSize: 14 }} /> Target Price (24h)
          </Typography>
          <Box>
            <Typography variant="h4" fontWeight="bold" sx={{ color: qfd?.emerald ?? 'primary.main' }}>
              ${data.predicted_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}
            </Typography>
          </Box>
        </Box>

        <Stack direction="row" spacing={1} alignItems="center">
          <Chip
            label={`${isPositive ? '+' : ''}${(data.drift * 100).toFixed(2)}% Predicted Drift`}
            size="small"
            color={isPositive ? 'success' : 'error'}
            variant="outlined"
            sx={{ fontWeight: 'bold' }}
          />
        </Stack>

        <Box sx={{ bgcolor: alpha(theme.palette.background.paper, 0.5), p: 1.5, borderRadius: 1, border: `1px solid ${alpha(theme.palette.divider, 0.1)}` }}>
          <Typography variant="caption" color="text.secondary" gutterBottom>
            95% Confidence Interval
          </Typography>
          <Typography variant="body2" fontWeight="medium">
            ${data.confidence_interval[0].toFixed(2)} — ${data.confidence_interval[1].toFixed(2)}
          </Typography>
        </Box>

        <Divider sx={{ opacity: 0.1 }} />

        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Box>
            <Typography variant="caption" color="text.secondary" display="block">
              Model
            </Typography>
            <Typography variant="caption" fontWeight="bold">
              {data.model_name}
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'right' }}>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Update sx={{ fontSize: 12 }} /> Updated
            </Typography>
            <Typography variant="caption">
              {new Date(data.last_updated).toLocaleTimeString()}
            </Typography>
          </Box>
        </Stack>
      </Stack>
    </Box>
  );
});