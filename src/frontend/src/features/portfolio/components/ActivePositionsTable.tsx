import React from 'react';
import { Box, Typography, Table, TableBody, TableCell, TableHead, TableRow, alpha } from '@mui/material';
import { stitchTokens } from '../../../theme/stitch-tokens';

const positions = [
  { symbol: 'AAPL_240615_C_190/195', type: 'BULL_CALL_SPREAD', strike: '190.0/195.0', pl: '+$1,420.42', plPercent: '+24.5%', delta: '+45.2', theta: '-8.12', gamma: '+2.14' },
  { symbol: 'TSLA_240621_P_175/170', type: 'PUT_CREDIT_SPREAD', strike: '175.0/170.0', pl: '-$420.15', plPercent: '-12.2%', delta: '-32.5', theta: '-5.40', gamma: '+1.82' },
  { symbol: 'NVDA_240628_C_900/920', type: 'CALL_DEBIT_SPREAD', strike: '900.0/920.0', pl: '+$3,850.12', plPercent: '+45.8%', delta: '+62.1', theta: '-12.45', gamma: '+3.20' },
  { symbol: 'SPY_240719_C_520/525', type: 'BULL_CALL_SPREAD', strike: '520.0/525.0', pl: '+$842.05', plPercent: '+8.4%', delta: '+28.4', theta: '-4.20', gamma: '+1.15' },
];

export const ActivePositionsTable: React.FC = () => {
  return (
    <Box sx={{ width: '100%', overflowX: 'auto' }}>
      <Table size="small" sx={{ minWidth: 800 }}>
        <TableHead>
          <TableRow sx={{ bgcolor: 'rgba(255,255,255,0.02)' }}>
            <TableCell className="stitch-label" sx={{ py: 1.5, fontSize: '8px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>INSTRUMENT_ID</TableCell>
            <TableCell className="stitch-label" sx={{ py: 1.5, fontSize: '8px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>STRATEGY_v2</TableCell>
            <TableCell className="stitch-label" sx={{ py: 1.5, fontSize: '8px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>STRIKE_CONF_EXP</TableCell>
            <TableCell align="right" className="stitch-label" sx={{ py: 1.5, fontSize: '8px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>P&L_UNREALIZED_USD</TableCell>
            <TableCell align="right" className="stitch-label" sx={{ py: 1.5, fontSize: '8px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>GREEKS_SCAN (Δ/Θ/Γ)</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {positions.map((row, idx) => (
            <TableRow 
              key={idx} 
              sx={{ 
                borderBottom: '1px solid rgba(255,255,255,0.03)', 
                '&:hover': { bgcolor: 'rgba(255,255,255,0.02)' },
                transition: 'background-color 0.2s ease'
              }}
            >
              <TableCell sx={{ py: 1.2 }}>
                <Typography sx={{ fontWeight: 950, fontSize: '10px', color: '#fff', letterSpacing: '0.5px' }}>{row.symbol}</Typography>
              </TableCell>
              <TableCell sx={{ py: 1.2 }}>
                <Box sx={{ 
                  display: 'inline-block', 
                  p: '1px 8px', 
                  bgcolor: 'rgba(0, 255, 163, 0.05)', 
                  border: `1px solid ${alpha(stitchTokens.colors.primary, 0.2)}`
                }}>
                   <Typography sx={{ fontSize: '9px', color: stitchTokens.colors.primary, fontWeight: 900 }}>{row.type}</Typography>
                </Box>
              </TableCell>
              <TableCell sx={{ py: 1.2 }}>
                <Typography className="stitch-mono" sx={{ fontSize: '10px', fontWeight: 700 }}>{row.strike}</Typography>
              </TableCell>
              <TableCell align="right" sx={{ py: 1.2 }}>
                <Typography className="stitch-mono" sx={{ fontSize: '11px', fontWeight: 950, color: row.pl.startsWith('+') ? stitchTokens.colors.primary : '#ff2e7e' }}>
                  {row.pl} <Box component="span" sx={{ fontSize: '9px', fontWeight: 700, opacity: 0.6 }}>({row.plPercent})</Box>
                </Typography>
              </TableCell>
              <TableCell align="right" sx={{ py: 1.2 }}>
                <Typography className="stitch-mono" sx={{ fontSize: '10px', fontWeight: 800 }}>
                  <Box component="span" sx={{ color: stitchTokens.colors.primary }}>{row.delta}</Box>
                  <Box component="span" sx={{ mx: 0.5, opacity: 0.2 }}>//</Box>
                  <Box component="span" sx={{ color: '#ff2e7e' }}>{row.theta}</Box>
                  <Box component="span" sx={{ mx: 0.5, opacity: 0.2 }}>//</Box>
                  <Box component="span" sx={{ color: stitchTokens.colors.secondary }}>{row.gamma}</Box>
                </Typography>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Box>
  );
};
