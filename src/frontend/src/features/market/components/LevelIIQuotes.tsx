import React from 'react';

import { Box, Typography, Grid, alpha } from '@mui/material';
import { stitchTokens } from '../../../theme/stitch-tokens';

export const LevelIIQuotes: React.FC<{ symbol: string }> = ({ symbol }) => {
  return (
    <Box className="stitch-card" sx={{ height: '100%' }}>
      <Box className="stitch-slanted-header">Level II // {symbol}</Box>
      <Box sx={{ p: 1.5 }}>
        <Grid container spacing={1}>
          <Grid item xs={6}>
            <Typography className="stitch-label" sx={{ mb: 1, color: stitchTokens.colors.primary }}>Bids</Typography>
            {[...Array(8)].map((_, i) => (
              <Box key={i} sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5, borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                <Typography className="stitch-mono" sx={{ fontSize: '10px' }}>189.{-i * 5}</Typography>
                <Typography className="stitch-mono" sx={{ fontSize: '10px', color: stitchTokens.colors.primary }}>{100 - i * 10}</Typography>
              </Box>
            ))}
          </Grid>
          <Grid item xs={6}>
            <Typography className="stitch-label" sx={{ mb: 1, color: '#ff4d4d' }}>Asks</Typography>
            {[...Array(8)].map((_, i) => (
              <Box key={i} sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5, borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                <Typography className="stitch-mono" sx={{ fontSize: '10px' }}>189.{50 + i * 5}</Typography>
                <Typography className="stitch-mono" sx={{ fontSize: '10px', color: '#ff4d4d' }}>{120 + i * 15}</Typography>
              </Box>
            ))}
          </Grid>
        </Grid>
      </Box>
    </Box>
  );
};
