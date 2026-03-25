import React from 'react';
import { Box, Typography, Stack, Button, alpha } from '@mui/material';
import { stitchTokens } from '../../../theme/stitch-tokens';

export const BlackSwanStressTest: React.FC = () => {
  const scenarios = [
    { name: '2008 GFC Repeat', impact: '-18.4%', risk: 'CRITICAL' },
    { name: '1987 Black Monday', impact: '-22.1%', risk: 'FATAL' },
    { name: 'Covid-19 Flash Crash', impact: '-12.5%', risk: 'HIGH' },
    { name: 'Interest Rate Spike', impact: '-5.2%', risk: 'MODERATE' }
  ];

  return (
    <Box className="stitch-card" sx={{ height: '100%', p: 0 }}>
      <Box className="stitch-slanted-header" sx={{ bgcolor: '#ff4d4d' }}>BLACK SWAN STRESS TEST // SIMULATOR</Box>
      <Box sx={{ p: 2 }}>
        <Stack spacing={1.5}>
          {scenarios.map((s, i) => (
            <Box key={i} sx={{ p: 1.5, bgcolor: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
               <Stack direction="row" justifyContent="space-between" alignItems="center">
                  <Box>
                    <Typography sx={{ fontSize: '11px', fontWeight: 700 }}>{s.name}</Typography>
                    <Typography className="stitch-label" sx={{ fontSize: '8px', color: s.risk === 'FATAL' || s.risk === 'CRITICAL' ? '#ff4d4d' : '#ffa500' }}>
                      RISK LEVEL: {s.risk}
                    </Typography>
                  </Box>
                  <Typography className="stitch-mono" sx={{ fontSize: '14px', fontWeight: 900, color: '#ff4d4d' }}>
                    {s.impact}
                  </Typography>
               </Stack>
            </Box>
          ))}
        </Stack>
        <Button 
          fullWidth 
          sx={{ 
            mt: 2, 
            borderRadius: 0, 
            bgcolor: alpha('#ff4d4d', 0.1), 
            color: '#ff4d4d', 
            border: '1px solid #ff4d4d',
            fontWeight: 900,
            fontSize: '10px'
          }}
        >
          RUN FULL PORTFOLIO DESTRUCTION TEST
        </Button>
      </Box>
    </Box>
  );
};
