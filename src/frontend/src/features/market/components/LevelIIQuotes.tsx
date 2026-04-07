import React from 'react';
import { Box, Typography, Grid, Stack } from '@mui/material';
import { stitchTokens } from '../../../theme/stitch-tokens';

export const LevelIIQuotes: React.FC<{ symbol: string }> = ({ symbol }) => {
  return (
    <Box className="stitch-card" sx={{ height: '100%', display: 'flex', flexDirection: 'column', p: 0, position: 'relative' }}>
      <Box className="stitch-slanted-header" sx={{ bgcolor: stitchTokens.colors.secondary, borderBottom: 'none' }}>LEVEL_II_QUOTES // {symbol}</Box>
      <Box sx={{ p: 1.5, flexGrow: 1, overflow: 'auto' }}>
        <Grid container spacing={2}>
          <Grid sx={{ width: '50%' }}>
            <Typography className="stitch-label" sx={{ mb: 1, color: stitchTokens.colors.primary, fontSize: '9px', fontWeight: 900 }}>BIDS_VWAP</Typography>
            <Stack spacing={0.5}>
              {[...Array(10)].map((_, i) => (
                <Box key={i} sx={{ display: 'flex', justifyContent: 'space-between', p: '2px 0', borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                  <Typography className="stitch-mono" sx={{ fontSize: '10px', fontWeight: 700 }}>189.{(45 - i * 5).toString().padStart(2, '0')}</Typography>
                  <Typography className="stitch-mono" sx={{ fontSize: '10px', fontWeight: 900, color: stitchTokens.colors.primary }}>{800 - i * 45}</Typography>
                </Box>
              ))}
            </Stack>
          </Grid>
          <Grid sx={{ width: '50%' }}>
            <Typography className="stitch-label" sx={{ mb: 1, color: '#ff2e7e', fontSize: '9px', fontWeight: 900 }}>ASKS_VWAP</Typography>
            <Stack spacing={0.5}>
              {[...Array(10)].map((_, i) => (
                <Box key={i} sx={{ display: 'flex', justifyContent: 'space-between', p: '2px 0', borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                  <Typography className="stitch-mono" sx={{ fontSize: '10px', fontWeight: 700 }}>189.{(50 + i * 5).toString().padStart(2, '0')}</Typography>
                  <Typography className="stitch-mono" sx={{ fontSize: '10px', fontWeight: 900, color: '#ff2e7e' }}>{920 + i * 60}</Typography>
                </Box>
              ))}
            </Stack>
          </Grid>
        </Grid>
      </Box>
      <Box sx={{ p: '6px 12px', bgcolor: 'rgba(0,0,0,0.2)', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
         <Typography className="stitch-label" sx={{ fontSize: '7px', opacity: 0.6 }}>IMBALANCE: <Box component="span" sx={{ color: stitchTokens.colors.primary }}>+12.4% BUY_SIDE</Box></Typography>
      </Box>
    </Box>
  );
};
