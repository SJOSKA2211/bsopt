import React from 'react';
import { Box, Typography, Stack, Slider, TextField, MenuItem, alpha, Grid } from '@mui/material';
import { stitchTokens } from '../../../theme/stitch-tokens';

export const OptimizationControls: React.FC = () => {
  return (
    <Box>
      <Grid container spacing={4}>
        <Grid item xs={12} md={4}>
          <Typography className="stitch-label" sx={{ mb: 2 }}>STRIKE RANGE [$]</Typography>
          <Stack direction="row" justifyContent="space-between" sx={{ mb: 1 }}>
            <Typography className="stitch-mono" sx={{ fontSize: '10px' }}>180.0</Typography>
            <Typography className="stitch-mono" sx={{ fontSize: '10px' }}>200.0</Typography>
          </Stack>
          <Slider 
            defaultValue={[180, 200]} 
            min={120} 
            max={240} 
            step={2.5}
            sx={{ 
              color: stitchTokens.colors.primary,
              '& .MuiSlider-thumb': { borderRadius: 0, width: 12, height: 12 },
              '& .MuiSlider-track': { borderRadius: 0 },
              '& .MuiSlider-rail': { opacity: 0.1 }
            }} 
          />
        </Grid>

        <Grid item xs={12} md={4}>
          <Typography className="stitch-label" sx={{ mb: 2 }}>EXPIRY RANGE [DAYS]</Typography>
          <Stack direction="row" justifyContent="space-between" sx={{ mb: 1 }}>
            <Typography className="stitch-mono" sx={{ fontSize: '10px' }}>30</Typography>
            <Typography className="stitch-mono" sx={{ fontSize: '10px' }}>90</Typography>
          </Stack>
          <Slider 
            defaultValue={[30, 90]} 
            min={0} 
            max={120} 
            step={10}
            sx={{ 
              color: stitchTokens.colors.secondary,
              '& .MuiSlider-thumb': { borderRadius: 0, width: 12, height: 12 },
              '& .MuiSlider-track': { borderRadius: 0 }
            }} 
          />
        </Grid>

        <Grid item xs={12} md={4}>
          <Typography className="stitch-label" sx={{ mb: 1.5 }}>OPTIMIZATION GOAL</Typography>
          <TextField
              select
              fullWidth
              size="small"
              defaultValue="sharpe"
              variant="standard"
              InputProps={{
                disableUnderline: true,
                sx: { 
                  className: 'stitch-mono', 
                  fontSize: '11px', 
                  bgcolor: 'rgba(255,255,255,0.03)', 
                  p: '4px 12px',
                  fontWeight: 800
                }
              }}
            >
              <MenuItem value="sharpe">MAX_SHARPE_RATIO</MenuItem>
              <MenuItem value="profit">MAX_PROFIT_PROB</MenuItem>
              <MenuItem value="drawdown">MIN_MAX_DRAWDOWN</MenuItem>
            </TextField>
            <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: 1.5 }}>
              <Box sx={{ width: 6, height: 6, bgcolor: stitchTokens.colors.primary }} />
              <Typography className="stitch-label" sx={{ fontSize: '8px', opacity: 0.5 }}>MODEL_SUBSYSTEM_ONLINE</Typography>
            </Stack>
        </Grid>
      </Grid>
    </Box>
  );
};

import { Paper, Grid } from '@mui/material';
