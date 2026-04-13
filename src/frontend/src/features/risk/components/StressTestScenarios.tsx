import React from 'react';
import { Box, Typography, Stack, alpha, useTheme, Chip, Divider } from '@mui/material';

const scenarios = [
  { name: 'Standard Stress (SPX -10%)', impact: '-$24,500', pct: '-9.4%', status: 'Warning' },
  { name: 'Volatility Spike (+20% Vol)', impact: '+$8,200', pct: '+3.1%', status: 'Benefit' },
  { name: 'Flash Crash (SPX -25%)', impact: '-$85,000', pct: '-32.1%', status: 'Critical' },
  { name: 'Interest Rate Hike (+1.0%)', impact: '-$1,200', pct: '-0.4%', status: 'Neutral' },
];

export const StressTestScenarios: React.FC = () => {

  return (
    <Box>
      <Stack spacing={2}>
        {scenarios.map((scenario, idx) => (
          <Box key={idx}>
            <Stack direction="row" justifyContent="space-between" alignItems="center">
              <Box>
                <Typography variant="body2" sx={{ fontWeight: 800 }}>{scenario.name}</Typography>
                <Chip 
                  label={scenario.status} 
                  size="small" 
                  sx={{ 
                    height: 16, 
                    fontSize: '0.6rem', 
                    fontWeight: 800, 
                    mt: 0.5,
                    bgcolor: alpha(scenario.status === 'Critical' ? '#ff2e7e' : scenario.status === 'Warning' ? '#f59e0b' : '#00ffa3', 0.1),
                    color: scenario.status === 'Critical' ? '#ff2e7e' : scenario.status === 'Warning' ? '#f59e0b' : '#00ffa3',
                  }} 
                />
              </Box>
              <Box sx={{ textAlign: 'right' }}>
                <Typography variant="body2" sx={{ fontWeight: 900, color: scenario.impact.startsWith('-') ? 'error.main' : 'success.main', fontFamily: 'JetBrains Mono' }}>
                  {scenario.impact}
                </Typography>
                <Typography variant="caption" sx={{ color: 'text.disabled', fontWeight: 700 }}>{scenario.pct}</Typography>
              </Box>
            </Stack>
            <Divider sx={{ mt: 1.5, borderColor: alpha('#fff', 0.03) }} />
          </Box>
        ))}
      </Stack>
    </Box>
  );
};
