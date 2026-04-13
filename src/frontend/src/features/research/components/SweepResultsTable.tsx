import React from 'react';
import { Box, Table, TableBody, TableCell, TableHead, TableRow, alpha, useTheme, Typography } from '@mui/material';
import { stitchTokens } from '../../../theme/stitch-tokens';

const sweepData = [
  { id: 'SWEEP_081', strikeShort: 190.0, strikeLong: 195.0, expiry: '45_DAY', profitProb: '68.52%', maxLoss: '-$210.42', delta: '+8.14', theta: '-0.08', gamma: '+0.02', score: 98.2 },
  { id: 'SWEEP_082', strikeShort: 187.5, strikeLong: 192.5, expiry: '45_DAY', profitProb: '65.24%', maxLoss: '-$185.15', delta: '+7.25', theta: '-0.06', gamma: '+0.01', score: 95.8 },
  { id: 'SWEEP_083', strikeShort: 192.5, strikeLong: 197.5, expiry: '45_DAY', profitProb: '61.85%', maxLoss: '-$240.10', delta: '+8.92', theta: '-0.10', gamma: '+0.03', score: 92.4 },
  { id: 'SWEEP_084', strikeShort: 190.0, strikeLong: 200.0, expiry: '45_DAY', profitProb: '58.12%', maxLoss: '-$320.05', delta: '+12.42', theta: '-0.14', gamma: '+0.05', score: 88.6 },
];

export const SweepResultsTable: React.FC = () => {
  return (
    <Box sx={{ width: '100%', overflowX: 'auto' }}>
      <Table size="small" sx={{ minWidth: 1000 }}>
        <TableHead>
          <TableRow sx={{ bgcolor: 'rgba(255,255,255,0.02)' }}>
            <TableCell className="stitch-label" sx={{ py: 1.5, fontSize: '8px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>CONFIG_ID</TableCell>
            <TableCell align="right" className="stitch-label" sx={{ py: 1.5, fontSize: '8px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>STRIKE_SHORT</TableCell>
            <TableCell align="right" className="stitch-label" sx={{ py: 1.5, fontSize: '8px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>STRIKE_LONG</TableCell>
            <TableCell align="right" className="stitch-label" sx={{ py: 1.5, fontSize: '8px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>EXP_EXPIRY</TableCell>
            <TableCell align="right" className="stitch-label" sx={{ py: 1.5, fontSize: '8px', color: stitchTokens.colors.primary, borderBottom: '1px solid rgba(255,255,255,0.05)' }}>PROB_PROFIT</TableCell>
            <TableCell align="right" className="stitch-label" sx={{ py: 1.5, fontSize: '8px', color: '#ff2e7e', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>MAX_LOSS_USD</TableCell>
            <TableCell align="right" className="stitch-label" sx={{ py: 1.5, fontSize: '8px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>Δ_DELTA</TableCell>
            <TableCell align="right" className="stitch-label" sx={{ py: 1.5, fontSize: '8px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>Θ_THETA</TableCell>
            <TableCell align="right" className="stitch-label" sx={{ py: 1.5, fontSize: '8px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>Γ_GAMMA</TableCell>
            <TableCell align="right" className="stitch-label" sx={{ py: 1.5, fontSize: '8px', color: stitchTokens.colors.primary, borderBottom: '1px solid rgba(255,255,255,0.05)' }}>HEURISTIC_SCORE</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {sweepData.map((row, idx) => (
            <TableRow 
              key={idx} 
              sx={{ 
                borderBottom: '1px solid rgba(255,255,255,0.03)', 
                '&:hover': { bgcolor: 'rgba(255,255,255,0.02)' },
                transition: 'background-color 0.2s ease'
              }}
            >
              <TableCell sx={{ py: 1.2 }}>
                <Typography sx={{ fontWeight: 950, fontSize: '10px', color: 'rgba(255,255,255,0.4)', fontFamily: 'JetBrains Mono' }}>{row.id}</Typography>
              </TableCell>
              <TableCell align="right" sx={{ py: 1.2 }}>
                <Typography className="stitch-mono" sx={{ fontSize: '11px', fontWeight: 900, color: stitchTokens.colors.primary }}>{row.strikeShort.toFixed(1)}</Typography>
              </TableCell>
              <TableCell align="right" sx={{ py: 1.2 }}>
                <Typography className="stitch-mono" sx={{ fontSize: '11px', fontWeight: 900, color: stitchTokens.colors.secondary }}>{row.strikeLong.toFixed(1)}</Typography>
              </TableCell>
              <TableCell align="right" sx={{ py: 1.2 }}>
                <Typography sx={{ fontSize: '10px', fontWeight: 700, opacity: 0.6 }}>{row.expiry}</Typography>
              </TableCell>
              <TableCell align="right" sx={{ py: 1.2, bgcolor: alpha(stitchTokens.colors.primary, 0.05) }}>
                <Typography className="stitch-mono" sx={{ fontSize: '11px', fontWeight: 950, color: stitchTokens.colors.primary }}>{row.profitProb}</Typography>
              </TableCell>
              <TableCell align="right" sx={{ py: 1.2 }}>
                <Typography className="stitch-mono" sx={{ fontSize: '11px', fontWeight: 950, color: '#ff2e7e' }}>{row.maxLoss}</Typography>
              </TableCell>
              <TableCell align="right" sx={{ py: 1.2 }}>
                <Typography className="stitch-mono" sx={{ fontSize: '10px', fontWeight: 800 }}>{row.delta}</Typography>
              </TableCell>
              <TableCell align="right" sx={{ py: 1.2 }}>
                <Typography className="stitch-mono" sx={{ fontSize: '10px', fontWeight: 800, color: '#ff2e7e' }}>{row.theta}</Typography>
              </TableCell>
              <TableCell align="right" sx={{ py: 1.2 }}>
                <Typography className="stitch-mono" sx={{ fontSize: '10px', fontWeight: 800 }}>{row.gamma}</Typography>
              </TableCell>
              <TableCell align="right" sx={{ py: 1.2, borderLeft: '1px solid rgba(255,255,255,0.05)' }}>
                <Typography className="stitch-mono" sx={{ fontSize: '14px', fontWeight: 950, color: stitchTokens.colors.primary }}>{row.score.toFixed(1)}</Typography>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Box>
  );
};
