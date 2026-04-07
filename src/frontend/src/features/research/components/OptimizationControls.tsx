import React from 'react';
import { Box, Typography, Stack, Slider, TextField, MenuItem, alpha, Grid } from '@mui/material';
import { stitchTokens } from '../../../theme/stitch-tokens';

export const OptimizationControls: React.FC = () => {
  return (
    <Box sx={{ position: 'relative' }}>
      <Grid container spacing={4} sx={{ width: '100%' }}>
        <Grid sx={{ width: { xs: '100%', md: '33.333333%' } }}>
          <Typography className="stitch-label" sx={{ mb: 2, fontSize: '9px', fontWeight: 900, letterSpacing: '1px' }}>STRIKE_RANGE_SENSITIVITY [$]</Typography>
          <Stack direction="row" justifyContent="space-between" sx={{ mb: 1.5 }}>
            <Typography className="stitch-mono" sx={{ fontSize: '10px', fontWeight: 900 }}>180.0</Typography>
            <Typography className="stitch-mono" sx={{ fontSize: '10px', fontWeight: 900, color: stitchTokens.colors.primary }}>200.0</Typography>
          </Stack>
          <Slider 
            defaultValue={[180, 200]} 
            min={120} 
            max={240} 
            step={2.5}
            sx={{ 
              height: 2,
              color: stitchTokens.colors.primary,
              '& .MuiSlider-thumb': { 
                borderRadius: 0, 
                width: 14, 
                height: 14, 
                bgcolor: '#000', 
                border: `2px solid ${stitchTokens.colors.primary}`,
                '&:hover': { boxShadow: `0 0 10px ${alpha(stitchTokens.colors.primary, 0.5)}` }
              },
              '& .MuiSlider-track': { borderRadius: 0 },
              '& .MuiSlider-rail': { opacity: 0.1, bgcolor: '#fff' }
            }} 
          />
        </Grid>

        <Grid sx={{ width: { xs: '100%', md: '33.333333%' } }}>
          <Typography className="stitch-label" sx={{ mb: 2, fontSize: '9px', fontWeight: 900, letterSpacing: '1px' }}>EXPIRY_HORIZON_SCAN [DAYS]</Typography>
          <Stack direction="row" justifyContent="space-between" sx={{ mb: 1.5 }}>
            <Typography className="stitch-mono" sx={{ fontSize: '10px', fontWeight: 900 }}>30</Typography>
            <Typography className="stitch-mono" sx={{ fontSize: '10px', fontWeight: 900, color: stitchTokens.colors.secondary }}>90</Typography>
          </Stack>
          <Slider 
            defaultValue={[30, 90]} 
            min={0} 
            max={120} 
            step={10}
            sx={{ 
              height: 2,
              color: stitchTokens.colors.secondary,
              '& .MuiSlider-thumb': { 
                borderRadius: 0, 
                width: 14, 
                height: 14, 
                bgcolor: '#000', 
                border: `2px solid ${stitchTokens.colors.secondary}`,
                '&:hover': { boxShadow: `0 0 10px ${alpha(stitchTokens.colors.secondary, 0.5)}` }
              },
              '& .MuiSlider-track': { borderRadius: 0 },
              '& .MuiSlider-rail': { opacity: 0.1, bgcolor: '#fff' }
            }} 
          />
        </Grid>

        <Grid sx={{ width: { xs: '100%', md: '33.333333%' } }}>
          <Typography className="stitch-label" sx={{ mb: 2, fontSize: '9px', fontWeight: 900, letterSpacing: '1px' }}>HEURISTIC_TARGET_OBJECTIVE</Typography>
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
                  bgcolor: 'rgba(255,255,255,0.02)', 
                  border: '1px solid rgba(255,255,255,0.05)',
                  p: '6px 16px',
                  fontWeight: 900,
                  color: stitchTokens.colors.primary
                }
              }}
            >
              <MenuItem value="sharpe">MAX_SHARPE_RATIO_v2</MenuItem>
              <MenuItem value="profit">MAX_PROB_PROFIT_SIGMA</MenuItem>
              <MenuItem value="drawdown">MIN_MAX_EXPECTED_DRAWDOWN</MenuItem>
            </TextField>
            <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mt: 2 }}>
              <Box sx={{ width: 8, height: 8, bgcolor: stitchTokens.colors.primary, boxShadow: `0 0 8px ${stitchTokens.colors.primary}` }} />
              <Typography className="stitch-label" sx={{ fontSize: '8px', fontWeight: 900, letterSpacing: '1px', opacity: 0.4 }}>CORE_ML_ENGINE_STABLE // READY_FOR_SWEEP</Typography>
            </Stack>
        </Grid>
      </Grid>
    </Box>
  );
};
