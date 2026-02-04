import React, { useMemo } from 'react';
import {
  Box,
  Typography,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Grid,
  Card,
  CardContent,
  alpha,
  useTheme,
  Chip,
  Tooltip,
} from '@mui/material';
import { usePortfolio } from '../hooks/usePortfolio';
import { useWasmPricing } from '../../../hooks/useWasmPricing';

const formatCurrency = (value: number) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(value);
};

export const PositionsSummary: React.FC = React.memo(() => {
  const theme = useTheme();
  const { data, isLoading, error } = usePortfolio();
  const { batchCalculate, isLoaded: isWasmLoaded } = useWasmPricing();

  const enrichedPositions = useMemo(() => {
    if (!data || !isWasmLoaded) return data?.positions || [];

    const now = new Date();
    const rate = 0.05;
    const div = 0.0;

    const params = data.positions.map(pos => {
      // Assuming pos has expiry and strike
      const expiryDate = pos.expiry ? new Date(pos.expiry) : new Date(now.getTime() + 30 * 24 * 60 * 60 * 1000);
      const timeToExpiry = Math.max(0.001, (expiryDate.getTime() - now.getTime()) / (1000 * 60 * 60 * 24 * 365));
      
      return {
        spot: pos.underlying_price || 150, // Fallback if missing
        strike: pos.strike || 150,
        time: timeToExpiry,
        vol: pos.implied_volatility || 0.2,
        rate,
        div,
        is_call: pos.option_type === 'call'
      };
    });

    const results = batchCalculate(params);

    return data.positions.map((pos, i) => ({
      ...pos,
      theor_greeks: results[i]?.greeks
    }));
  }, [data, isWasmLoaded, batchCalculate]);

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', p: 4 }}>
        <CircularProgress aria-label="Loading active positions" />
      </Box>
    );
  }

  if (error || !data) {
    return (
      <Box sx={{ p: 4, textAlign: 'center' }}>
        <Typography color="error">Error loading portfolio data</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ p: 2 }}>
        <Grid container spacing={2}>
          <Grid size={{ xs: 4 }}>
            <Card sx={{ bgcolor: alpha(theme.palette.primary.main, 0.05), boxShadow: 'none' }}>
              <CardContent sx={{ p: '16px !important' }}>
                <Typography variant="overline" color="text.secondary">Total Balance</Typography>
                <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
                  {formatCurrency(data.balance)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid size={{ xs: 4 }}>
            <Card sx={{ bgcolor: alpha(theme.palette.secondary.main, 0.05), boxShadow: 'none' }}>
              <CardContent sx={{ p: '16px !important' }}>
                <Typography variant="overline" color="text.secondary">Frozen Capital</Typography>
                <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
                  {formatCurrency(data.frozen_capital)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid size={{ xs: 4 }}>
            <Card sx={{ bgcolor: alpha(theme.palette.warning.main, 0.05), boxShadow: 'none' }}>
              <CardContent sx={{ p: '16px !important' }}>
                <Typography variant="overline" color="text.secondary">Risk Score</Typography>
                <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
                  {data.risk_score.toFixed(2)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>

      <Stack direction="row" alignItems="center" sx={{ px: 2, pb: 1, pt: 2 }} spacing={2}>
        <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
          Active Positions
        </Typography>
        {isWasmLoaded && (
          <Chip label="WASM Risk Engine Active" size="small" color="success" variant="outlined" sx={{ height: 20, fontSize: '0.65rem' }} />
        )}
      </Stack>

      <TableContainer component={Box} sx={{ flex: 1, overflow: 'auto' }}>
        <Table stickyHeader size="small">
          <TableHead>
            <TableRow>
              <TableCell sx={{ fontWeight: 'bold' }}>Symbol</TableCell>
              <TableCell align="right" sx={{ fontWeight: 'bold' }}>Quantity</TableCell>
              <TableCell align="right" sx={{ fontWeight: 'bold' }}>Delta (Δ)</TableCell>
              <TableCell align="right" sx={{ fontWeight: 'bold' }}>Gamma (Γ)</TableCell>
              <TableCell align="right" sx={{ fontWeight: 'bold' }}>Current P&L</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {enrichedPositions.map((pos) => (
              <TableRow key={pos.contract_symbol} hover>
                <TableCell>
                  <Typography variant="body2" sx={{ fontWeight: 'medium' }}>{pos.contract_symbol}</Typography>
                  <Typography variant="caption" color="text.secondary">{pos.option_type?.toUpperCase()} ${pos.strike} {pos.expiry}</Typography>
                </TableCell>
                <TableCell align="right">{pos.quantity}</TableCell>
                <TableCell align="right">
                  {pos.theor_greeks ? (
                    <Tooltip title="Theoretical Delta (WASM)">
                      <Typography variant="body2">{(pos.theor_greeks.delta * pos.quantity).toFixed(2)}</Typography>
                    </Tooltip>
                  ) : '---'}
                </TableCell>
                <TableCell align="right">
                  {pos.theor_greeks ? (
                    <Tooltip title="Theoretical Gamma (WASM)">
                      <Typography variant="body2">{(pos.theor_greeks.gamma * pos.quantity).toFixed(4)}</Typography>
                    </Tooltip>
                  ) : '---'}
                </TableCell>
                <TableCell 
                  align="right" 
                  sx={{ 
                    color: pos.current_pnl >= 0 ? 'success.main' : 'error.main',
                    fontWeight: 'medium'
                  }}
                >
                  {pos.current_pnl >= 0 ? '+' : ''}{formatCurrency(pos.current_pnl)}
                </TableCell>
              </TableRow>
            ))}
            {data.positions.length === 0 && (
              <TableRow>
                <TableCell colSpan={5} align="center" sx={{ py: 4 }}>
                  <Typography color="text.secondary">No active positions</Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
});
