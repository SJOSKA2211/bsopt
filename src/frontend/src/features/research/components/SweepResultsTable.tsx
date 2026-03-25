import React from 'react';
import { Box, Table, TableBody, TableCell, TableHead, TableRow, alpha, useTheme, Chip } from '@mui/material';

const sweepData = [
  { id: '#081', strikeShort: 190.0, strikeLong: 195.0, expiry: '45 Days', profitProb: '68.5%', maxLoss: '-$210', delta: 8.14, theta: -0.08, gamma: 0.02, score: 98.2 },
  { id: '#082', strikeShort: 187.5, strikeLong: 192.5, expiry: '45 Days', profitProb: '65.2%', maxLoss: '-$185', delta: 7.25, theta: -0.06, gamma: 0.01, score: 95.8 },
  { id: '#083', strikeShort: 192.5, strikeLong: 197.5, expiry: '45 Days', profitProb: '61.8%', maxLoss: '-$240', delta: 8.92, theta: -0.10, gamma: 0.03, score: 92.4 },
];

export const SweepResultsTable: React.FC = () => {
  const theme = useTheme();

  return (
    <Box sx={{ mt: 3 }}>
      <Table size="small">
        <TableHead>
          <TableRow sx={{ bgcolor: alpha('#000', 0.2) }}>
            <TableCell sx={{ color: 'text.disabled', border: 'none', fontSize: '0.65rem', fontWeight: 800 }}>CONFIG ID</TableCell>
            <TableCell align="right" sx={{ color: 'text.disabled', border: 'none', fontSize: '0.65rem', fontWeight: 800 }}>STRIKE SHORT</TableCell>
            <TableCell align="right" sx={{ color: 'text.disabled', border: 'none', fontSize: '0.65rem', fontWeight: 800 }}>STRIKE LONG</TableCell>
            <TableCell align="right" sx={{ color: 'text.disabled', border: 'none', fontSize: '0.65rem', fontWeight: 800 }}>EXPIRY</TableCell>
            <TableCell align="right" sx={{ color: 'success.main', border: 'none', fontSize: '0.65rem', fontWeight: 800 }}>PROFIT PROB.</TableCell>
            <TableCell align="right" sx={{ color: 'error.main', border: 'none', fontSize: '0.65rem', fontWeight: 800 }}>MAX LOSS</TableCell>
            <TableCell align="right" sx={{ color: 'text.disabled', border: 'none', fontSize: '0.65rem', fontWeight: 800 }}>DELTA</TableCell>
            <TableCell align="right" sx={{ color: 'text.disabled', border: 'none', fontSize: '0.65rem', fontWeight: 800 }}>THETA</TableCell>
            <TableCell align="right" sx={{ color: 'text.disabled', border: 'none', fontSize: '0.65rem', fontWeight: 800 }}>GAMMA</TableCell>
            <TableCell align="right" sx={{ color: 'primary.main', border: 'none', fontSize: '0.65rem', fontWeight: 800 }}>SCORE</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {sweepData.map((row, idx) => (
            <TableRow key={idx} sx={{ borderBottom: `1px solid ${alpha('#fff', 0.05)}`, '&:hover': { bgcolor: alpha('#fff', 0.01) } }}>
              <TableCell sx={{ border: 'none', py: 1.5, color: 'text.secondary', fontWeight: 700, fontSize: '0.75rem', fontFamily: 'JetBrains Mono' }}>{row.id}</TableCell>
              <TableCell align="right" sx={{ border: 'none', color: 'primary.main', fontWeight: 800, fontFamily: 'JetBrains Mono' }}>{row.strikeShort.toFixed(1)}</TableCell>
              <TableCell align="right" sx={{ border: 'none', color: 'secondary.main', fontWeight: 800, fontFamily: 'JetBrains Mono' }}>{row.strikeLong.toFixed(1)}</TableCell>
              <TableCell align="right" sx={{ border: 'none', fontWeight: 700 }}>{row.expiry}</TableCell>
              <TableCell align="right" sx={{ border: 'none', color: 'success.main', fontWeight: 800, bgcolor: alpha(theme.palette.success.main, 0.05) }}>{row.profitProb}</TableCell>
              <TableCell align="right" sx={{ border: 'none', color: 'error.main', fontWeight: 800 }}>{row.maxLoss}</TableCell>
              <TableCell align="right" sx={{ border: 'none', fontWeight: 700, fontFamily: 'JetBrains Mono' }}>{row.delta}</TableCell>
              <TableCell align="right" sx={{ border: 'none', color: 'error.main', fontWeight: 700, fontFamily: 'JetBrains Mono' }}>{row.theta}</TableCell>
              <TableCell align="right" sx={{ border: 'none', fontWeight: 700, fontFamily: 'JetBrains Mono' }}>{row.gamma}</TableCell>
              <TableCell align="right" sx={{ border: 'none', color: 'primary.main', fontWeight: 900, fontSize: '0.9rem', borderLeft: `1px solid ${alpha('#fff', 0.05)}` }}>{row.score}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Box>
  );
};
