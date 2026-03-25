import React from 'react';
import { Box, Typography, Stack, alpha } from '@mui/material';
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
      p: '16px 20px', 
      bgcolor: 'rgba(255,255,255,0.02)', 
      border: '1px solid rgba(255,255,255,0.05)',
      position: 'relative',
      overflow: 'hidden',
      transition: 'all 0.2s ease',
      cursor: 'pointer',
      '&:hover': {
        border: `1px solid ${alpha(stitchTokens.colors.primary, 0.3)}`,
        bgcolor: 'rgba(255,255,255,0.04)',
        '& .id-tag': { bgcolor: stitchTokens.colors.primary, color: '#000' }
      }
    }}>
      <Box className="stitch-dots-container" sx={{ opacity: 0.05 }} />
      
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1.5, position: 'relative', zIndex: 1 }}>
        <Box className="id-tag" sx={{ 
          px: 1, 
          py: 0.2, 
          bgcolor: 'rgba(255,255,255,0.05)', 
          border: '1px solid rgba(255,255,255,0.1)',
          transition: 'all 0.2s ease'
        }}>
           <Typography className="stitch-mono" sx={{ fontSize: '9px', fontWeight: 950 }}>
              #{id.toString().padStart(2, '0')}
           </Typography>
        </Box>
        <Box sx={{ px: 1, py: 0.2, bgcolor: alpha(stitchTokens.colors.primary, 0.1), borderLeft: `2px solid ${stitchTokens.colors.primary}` }}>
           <Typography className="stitch-mono" sx={{ fontSize: '9px', fontWeight: 950, color: stitchTokens.colors.primary }}>
              SCORE_{score}
           </Typography>
        </Box>
      </Stack>

      <Typography className="stitch-mono" sx={{ fontSize: '15px', fontWeight: 950, mb: 1, color: '#fff', position: 'relative', zIndex: 1 }}>
        {strike1.toFixed(1)} <Box component="span" sx={{ opacity: 0.2 }}>//</Box> {strike2.toFixed(1)}
      </Typography>

      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ position: 'relative', zIndex: 1 }}>
        <Typography className="stitch-label" sx={{ fontSize: '8px', fontWeight: 900, letterSpacing: '0.5px', opacity: 0.4 }}>EXP_PROB_PROFIT_v4</Typography>
        <Typography className="stitch-mono" sx={{ color: stitchTokens.colors.primary, fontWeight: 950, fontSize: '11px' }}>
          {change}
        </Typography>
      </Stack>
      
      {/* Abstract Geometric Decoration */}
      <Box className="stitch-abstract-shard" sx={{ position: 'absolute', bottom: -15, right: -15, width: 40, height: 40, bgcolor: alpha(stitchTokens.colors.primary, 0.02), clipPath: stitchTokens.geometry.shard }} />
    </Box>
  );
};
