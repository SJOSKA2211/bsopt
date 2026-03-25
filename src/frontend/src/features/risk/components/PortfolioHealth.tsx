import React from 'react';
import { Box, Typography, Stack, alpha } from '@mui/material';
import { stitchTokens } from '../../../theme/stitch-tokens';

export const PortfolioHealth: React.FC = () => {
  return (
    <Box className="stitch-card" sx={{ p: 3, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
       <Box>
          <Typography className="stitch-label" sx={{ mb: 1 }}>SYSTEMIC PORTFOLIO HEALTH</Typography>
          <Typography variant="h3" sx={{ fontWeight: 900, color: stitchTokens.colors.primary }}>
            92.4 <Typography component="span" variant="h6" sx={{ color: '#a9abb1', opacity: 0.5 }}>/ 100</Typography>
          </Typography>
          <Typography sx={{ fontSize: '10px', fontWeight: 500, color: stitchTokens.colors.primary, mt: 0.5 }}>
            Optimal Efficiency Mode Active
          </Typography>
       </Box>
       
       <Box sx={{ position: 'relative', width: 80, height: 80, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Box sx={{ 
            position: 'absolute', 
            width: '100%', 
            height: '100%', 
            borderRadius: '50%', 
            border: '4px solid rgba(255,255,255,0.05)',
          }} />
          <Box sx={{ 
            position: 'absolute', 
            width: '100%', 
            height: '100%', 
            borderRadius: '50%', 
            border: `4px solid ${stitchTokens.colors.primary}`,
            clipPath: 'polygon(50% 50%, 50% 0%, 100% 0%, 100% 100%, 0% 100%, 0% 0%, 50% 0%)',
            transform: 'rotate(45deg)'
          }} />
          <Typography className="stitch-mono" sx={{ fontSize: '12px', fontWeight: 900 }}>v4.0</Typography>
       </Box>
    </Box>
  );
};
