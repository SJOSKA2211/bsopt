import React from 'react';
import {
  Box,
  Typography,
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
} from '@mui/material';
import { usePortfolio } from '../hooks/usePortfolio';

const formatCurrency = (value: number) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(value);
};

export const PositionsSummary: React.FC = React.memo(() => {
  const theme = useTheme();
  const { data, isLoading, error } = usePortfolio();

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

      <Typography variant="h6" sx={{ px: 2, pb: 1, pt: 2, fontWeight: 'bold' }}>
        Active Positions
      </Typography>

      <TableContainer component={Box} sx={{ flex: 1, overflow: 'auto' }}>
        <Table stickyHeader size="small">
          <TableHead>
            <TableRow>
              <TableCell sx={{ fontWeight: 'bold' }}>Symbol</TableCell>
              <TableCell align="right" sx={{ fontWeight: 'bold' }}>Quantity</TableCell>
              <TableCell align="right" sx={{ fontWeight: 'bold' }}>Current P&L</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {data.positions.map((pos) => (
              <TableRow key={pos.contract_symbol} hover>
                <TableCell>{pos.contract_symbol}</TableCell>
                <TableCell align="right">{pos.quantity}</TableCell>
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
                <TableCell colSpan={3} align="center" sx={{ py: 4 }}>
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
