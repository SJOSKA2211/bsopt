import React from 'react';
import { Box, Typography, Stack, LinearProgress, alpha } from '@mui/material';
import { stitchTokens } from '../../../theme/stitch-tokens';

interface MetricProps {
  label: string;
  value: string;
  percent: number;
  color: string;
}

const ModelMetric: React.FC<MetricProps> = ({ label, value, percent, color }) => (
  <Box sx={{ mb: 2 }}>
    <Stack direction="row" justifyContent="space-between" sx={{ mb: 0.5 }}>
      <Typography className="stitch-label" sx={{ fontSize: '9px' }}>{label}</Typography>
      <Typography className="stitch-mono" sx={{ fontSize: '11px', fontWeight: 800, color }}>{value}</Typography>
    </Stack>
    <LinearProgress 
      variant="determinate" 
      value={percent} 
      aria-label={label}
      sx={{ 
        height: 4, 
        bgcolor: alpha(color, 0.1),
        '& .MuiLinearProgress-bar': { bgcolor: color }
      }} 
    />
  </Box>
);

export const DeepInferenceEngine: React.FC = () => {
  return (
    <Box className="stitch-card" sx={{ height: '100%', p: 0 }}>
      <Box className="stitch-slanted-header">DEEP-INFERENCE ML ENGINE // v4.2</Box>
      <Box sx={{ p: 2 }}>
        <ModelMetric label="Directional Prob (Bullish)" value="68.4%" percent={68.4} color={stitchTokens.colors.primary} />
        <ModelMetric label="Vol Expansion Confidence" value="82.1%" percent={82.1} color={stitchTokens.colors.secondary} />
        <ModelMetric label="Signal Strength" value="High" percent={90} color={stitchTokens.colors.tertiary} />
        
        <Box sx={{ mt: 3, p: 1.5, bgcolor: alpha(stitchTokens.colors.primary, 0.05), borderLeft: `2px solid ${stitchTokens.colors.primary}` }}>
          <Typography className="stitch-label" sx={{ fontSize: '8px', mb: 0.5 }}>LATEST INSIGHT</Typography>
          <Typography sx={{ fontSize: '11px', fontWeight: 500, lineHeight: 1.4 }}>
            Gamma-weighted distribution suggests potential 1.2% mean reversion within 4 hours.
          </Typography>
        </Box>
      </Box>
    </Box>
  );
};
