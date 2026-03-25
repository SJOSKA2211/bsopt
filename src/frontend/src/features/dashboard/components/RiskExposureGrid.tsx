import React from 'react';
import { Box, Typography, alpha } from '@mui/material';
import { stitchTokens } from '../../../theme/stitch-tokens';

export const RiskExposureGrid: React.FC = () => {
  const data = [
    [0.1, -0.2, 0.4, 0.1],
    [0.3, 0.5, -0.1, 0.2],
    [-0.2, 0.4, 0.8, -0.3],
    [0.1, 0.1, 0.2, 0.5]
  ];

  return (
    <Box className="stitch-card" sx={{ height: '100%', p: 0 }}>
      <Box className="stitch-slanted-header" sx={{ bgcolor: stitchTokens.colors.secondary }}>RISK EXPOSURE MATRIX</Box>
      <Box sx={{ p: 1, display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 0.5, height: 'calc(100% - 32px)' }}>
        {data.flat().map((val, i) => (
          <Box 
            key={i} 
            sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              bgcolor: val > 0 ? alpha(stitchTokens.colors.primary, val ) : alpha('#ff4d4d', Math.abs(val)),
              border: '1px solid rgba(255,255,255,0.05)'
            }}
          >
            <Typography className="stitch-mono" sx={{ fontSize: '9px', fontWeight: 900, color: Math.abs(val) > 0.5 ? 'black' : 'white' }}>
              {(val * 10).toFixed(1)}k
            </Typography>
          </Box>
        ))}
      </Box>
    </Box>
  );
};
