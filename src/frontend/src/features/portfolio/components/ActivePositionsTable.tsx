import React from 'react';
import { Box, Typography, Table, TableBody, TableCell, TableHead, TableRow, alpha } from '@mui/material';
import { stitchTokens } from '../../../theme/stitch-tokens';

const positions = [
  { symbol: 'AAPL', type: 'Bull Call', strike: '190/195', expiry: '15-Jun-24', pl: '+$1,420', plPercent: '+24.5%', delta: 45.2, theta: -8.12, gamma: 2.1 },
  { symbol: 'TSLA', type: 'Put Spread', strike: '175/170', expiry: '21-Jun-24', pl: '-$420', plPercent: '-12.2%', delta: -32.5, theta: -5.40, gamma: 1.8 },
  { symbol: 'NVDA', type: 'Call Spread', strike: '900/920', expiry: '28-Jun-24', pl: '+$3,850', plPercent: '+45.8%', delta: 62.1, theta: -12.45, gamma: 3.2 },
];

export const ActivePositionsTable: React.FC = () => {
  return (
    <Box>
      <Table size="small">
        <TableHead>
          <TableRow sx={{ bgcolor: 'rgba(255,255,255,0.02)' }}>
            <TableCell className="stitch-label" sx={{ py: 1, fontSize: '8px' }}>SYMBOL</TableCell>
            <TableCell className="stitch-label" sx={{ py: 1, fontSize: '8px' }}>STRATEGY</TableCell>
            <TableCell className="stitch-label" sx={{ py: 1, fontSize: '8px' }}>STRIKE/EXP</TableCell>
            <TableCell align="right" className="stitch-label" sx={{ py: 1, fontSize: '8px' }}>P&L (UNREALIZED)</TableCell>
            <TableCell align="right" className="stitch-label" sx={{ py: 1, fontSize: '8px' }}>GREEKS (Δ/Θ)</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {positions.map((row, idx) => (
            <TableRow key={idx} sx={{ borderBottom: '1px solid rgba(255,255,255,0.03)', '&:hover': { bgcolor: 'rgba(255,255,255,0.01)' } }}>
              <TableCell sx={{ py: 1 }}>
                <Typography sx={{ fontWeight: 800, fontSize: '11px' }}>{row.symbol}</Typography>
              </TableCell>
              <TableCell sx={{ py: 1 }}>
                <Typography sx={{ fontSize: '10px', color: stitchTokens.colors.primary, fontWeight: 700 }}>{row.type}</Typography>
              </TableCell>
              <TableCell sx={{ py: 1 }}>
                <Typography className="stitch-mono" sx={{ fontSize: '10px' }}>{row.strike} <Box component="span" sx={{ opacity: 0.5 }}>[{row.expiry}]</Box></Typography>
              </TableCell>
              <TableCell align="right" sx={{ py: 1 }}>
                <Typography className="stitch-mono" sx={{ fontSize: '11px', fontWeight: 900, color: row.pl.startsWith('+') ? stitchTokens.colors.primary : '#ff4d4d' }}>
                  {row.pl} <Box component="span" sx={{ fontSize: '9px', opacity: 0.8 }}>({row.plPercent})</Box>
                </Typography>
              </TableCell>
              <TableCell align="right" sx={{ py: 1 }}>
                <Typography className="stitch-mono" sx={{ fontSize: '10px' }}>
                  <Box component="span" sx={{ color: stitchTokens.colors.primary }}>{row.delta}</Box>
                  <Box component="span" sx={{ mx: 0.5, opacity: 0.3 }}>|</Box>
                  <Box component="span" sx={{ color: '#ff4d4d' }}>{row.theta}</Box>
                </Typography>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Box>
  );
};
