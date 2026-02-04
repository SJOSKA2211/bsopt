import React from 'react';
import {
  Box,
  Typography,
  Stack,
  Divider,
  CircularProgress,
  useTheme,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  AccountBalanceWallet,
} from '@mui/icons-material';
import { usePortfolio } from '../hooks/usePortfolio';

export const PortfolioSummary: React.FC = React.memo(() => {
  const theme = useTheme();
  const { data, isLoading, isError } = usePortfolio();

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
        <CircularProgress size={40} aria-label="Loading portfolio summary" />
      </Box>
    );
  }

  if (isError || !data) {
    return (
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography color="error">Error loading portfolio data</Typography>
      </Box>
    );
  }

  const { totalValue, dailyPnL, dailyPnLPercent, positionsCount } = data;
  const isPositive = dailyPnL >= 0;

  return (
    <Box sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 3 }}>
        <AccountBalanceWallet color="primary" />
        <Typography variant="h6" fontWeight="bold">
          Portfolio Overview
        </Typography>
      </Stack>

      <Stack spacing={3}>
        <Box>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Total Portfolio Value
          </Typography>
          <Typography variant="h4" fontWeight="bold" color="text.primary">
            ${totalValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </Typography>
        </Box>

        <Divider sx={{ borderStyle: 'dashed', opacity: 0.5 }} />

        <Stack direction="row" spacing={4}>
          <Box sx={{ flex: 1 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Daily P&L
            </Typography>
            <Stack direction="row" spacing={0.5} alignItems="center">
              {isPositive ? (
                <TrendingUp fontSize="small" sx={{ color: theme.palette.success.main }} />
              ) : (
                <TrendingDown fontSize="small" sx={{ color: theme.palette.error.main }} />
              )}
              <Typography
                variant="h6"
                fontWeight="bold"
                sx={{ color: isPositive ? theme.palette.success.main : theme.palette.error.main }}
              >
                {isPositive ? '+' : ''}${Math.abs(dailyPnL).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </Typography>
            </Stack>
          </Box>

          <Box sx={{ flex: 1 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Return
            </Typography>
            <Typography
              variant="h6"
              fontWeight="bold"
              sx={{ color: isPositive ? theme.palette.success.main : theme.palette.error.main }}
            >
              {isPositive ? '+' : ''}{dailyPnLPercent.toFixed(2)}%
            </Typography>
          </Box>
        </Stack>

        <Divider sx={{ borderStyle: 'dashed', opacity: 0.5 }} />

        <Box sx={{ mt: 'auto' }}>
          <Typography variant="body2" color="text.secondary">
            Current Status
          </Typography>
          <Typography variant="body1" fontWeight="medium">
            {positionsCount} Positions
          </Typography>
        </Box>
      </Stack>
    </Box>
  );
});
