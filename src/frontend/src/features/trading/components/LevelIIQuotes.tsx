import React from 'react';
import { Box, Typography, Table, TableBody, TableCell, TableHead, TableRow, alpha } from '@mui/material';

const quotes = [
  { mmid: 'NSDQ', bid: 4.25, size: 10, ask: 4.26, askSize: 5 },
  { mmid: 'ARCA', bid: 4.25, size: 8, ask: 4.26, askSize: 12 },
  { mmid: 'BATS', bid: 4.24, size: 20, ask: 4.27, askSize: 2 },
  { mmid: 'EDGX', bid: 4.24, size: 5, ask: 4.27, askSize: 30 },
  { mmid: 'IEX', bid: 4.23, size: 15, ask: 4.28, askSize: 1 },
  { mmid: 'AMEX', bid: 4.23, size: 12, ask: 4.28, askSize: 8 },
  { mmid: 'PHLX', bid: 4.22, size: 50, ask: 4.29, askSize: 15 },
  { mmid: 'NYSE', bid: 4.21, size: 100, ask: 4.30, askSize: 22 },
];

export const LevelIIQuotes: React.FC = () => {

  return (
    <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 280 }}>
      {/* Header */}
      <Box sx={{ p: 2, display: 'flex', alignItems: 'center', gap: 1, borderBottom: `1px solid ${alpha('#fff', 0.05)}` }}>
        <Typography variant="caption" sx={{ fontSize: '0.7rem', fontWeight: 800, color: 'text.secondary' }}>LEVEL II QUOTES</Typography>
      </Box>

      {/* Table */}
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell sx={{ color: 'text.disabled', borderBottom: `1px solid ${alpha('#fff', 0.05)}`, fontSize: '0.62rem', fontWeight: 700 }}>MMID</TableCell>
            <TableCell align="right" sx={{ color: 'success.main', borderBottom: `1px solid ${alpha('#fff', 0.05)}`, fontSize: '0.62rem', fontWeight: 700 }}>BID</TableCell>
            <TableCell align="right" sx={{ color: 'text.disabled', borderBottom: `1px solid ${alpha('#fff', 0.05)}`, fontSize: '0.62rem', fontWeight: 700 }}>SIZE</TableCell>
            <TableCell align="right" sx={{ color: 'error.main', borderBottom: `1px solid ${alpha('#fff', 0.05)}`, fontSize: '0.62rem', fontWeight: 700 }}>ASK</TableCell>
            <TableCell align="right" sx={{ color: 'text.disabled', borderBottom: `1px solid ${alpha('#fff', 0.05)}`, fontSize: '0.62rem', fontWeight: 700 }}>SIZE</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {quotes.map((row, idx) => (
            <TableRow key={idx} sx={{ '&:hover': { bgcolor: alpha('#fff', 0.02) } }}>
              <TableCell sx={{ border: 'none', py: 1.2, fontWeight: 700, fontSize: '0.75rem', color: 'text.secondary' }}>{row.mmid}</TableCell>
              <TableCell align="right" sx={{ border: 'none', py: 1.2, fontWeight: 800, color: 'success.main', fontFamily: 'JetBrains Mono' }}>{row.bid.toFixed(2)}</TableCell>
              <TableCell align="right" sx={{ border: 'none', py: 1.2, fontWeight: 600, color: 'text.primary', fontFamily: 'JetBrains Mono' }}>{row.size}</TableCell>
              <TableCell align="right" sx={{ border: 'none', py: 1.2, fontWeight: 800, color: 'error.main', fontFamily: 'JetBrains Mono' }}>{row.ask.toFixed(2)}</TableCell>
              <TableCell align="right" sx={{ border: 'none', py: 1.2, fontWeight: 600, color: 'text.primary', fontFamily: 'JetBrains Mono' }}>{row.askSize}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Box>
  );
};
