import React from 'react';
import { Box, Typography, Stack, Divider } from '@mui/material';
import { stitchTokens } from '../../../theme/stitch-tokens';

interface OptimalConfigCardProps {
  id: number;
  strike1: number;
  strike2: number;
  change: string;
  score: string;
}

export const OptimalConfigCard: React.FC<OptimalConfigCardProps> = ({ id, strike1, strike2, change, score }) => {
  return (
    <Box sx={{ 
      p: 1.5, 
      bgcolor: 'rgba(255,255,255,0.02)', 
      border: '1px solid rgba(255,255,255,0.05)',
      position: 'relative',
      '&:hover': {
        border: `1px solid ${stitchTokens.colors.primary}44`,
        bgcolor: 'rgba(255,255,255,0.04)',
      }
    }}>
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
        <Typography className="stitch-mono" sx={{ color: stitchTokens.colors.primary, fontSize: '10px', fontWeight: 900 }}>
          #{id.toString().padStart(2, '0')}
        </Typography>
        <Box sx={{ px: 1, py: 0.2, bgcolor: `${stitchTokens.colors.primary}22`, borderRadius: 0 }}>
           <Typography className="stitch-mono" sx={{ fontSize: '9px', fontWeight: 900, color: stitchTokens.colors.primary }}>
              SCORE_{score}
           </Typography>
        </Box>
      </Stack>

      <Typography className="stitch-mono" sx={{ fontSize: '14px', fontWeight: 900, mb: 1 }}>
        {strike1} <Box component="span" sx={{ opacity: 0.2 }}>/</Box> {strike2}
      </Typography>

      <Divider sx={{ mb: 1, borderColor: 'rgba(255,255,255,0.03)' }} />
      
      <Stack direction="row" justifyContent="space-between" alignItems="center">
        <Typography className="stitch-label" sx={{ fontSize: '8px' }}>PROB_PROFIT</Typography>
        <Typography className="stitch-mono" sx={{ color: stitchTokens.colors.primary, fontWeight: 900, fontSize: '10px' }}>
          {change}
        </Typography>
      </Stack>
    </Box>
  );
};
