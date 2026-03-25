import React from 'react';
import { Box, Typography, alpha, useTheme } from '@mui/material';

const priceSteps = [-10, -5, -2, 0, 2, 5, 10];
const volSteps = [-5, -2, 0, 2, 5];

const generateData = () => {
  const data = [];
  for (const vol of volSteps) {
    const row = [];
    for (const price of priceSteps) {
      // Mock P&L calculation
      const val = (price * 2.5) - (vol * 1.8);
      row.push(val);
    }
    data.push(row);
  }
  return data;
};

export const RiskHeatmap: React.FC = () => {
  const theme = useTheme();
  const data = generateData();

  const getColor = (val: number) => {
    if (val > 0) return alpha(theme.palette.success.main, Math.min(Math.abs(val) / 20, 0.8));
    if (val < 0) return alpha(theme.palette.error.main, Math.min(Math.abs(val) / 20, 0.8));
    return alpha('#fff', 0.05);
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="caption" sx={{ color: 'text.disabled', mb: 3, display: 'block', textAlign: 'center' }}>
        P&L IMPACT: PRICE (%) vs VOLATILITY (%)
      </Typography>
      
      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(8, 1fr)', gap: 0.5 }}>
        {/* Placeholder for top-left corner */}
        <Box />
        {/* Price Headers */}
        {priceSteps.map(p => (
          <Box key={p} sx={{ textAlign: 'center', py: 1 }}>
            <Typography variant="caption" sx={{ fontWeight: 800, color: p === 0 ? 'primary.main' : 'text.disabled' }}>{p > 0 ? `+${p}` : p}%</Typography>
          </Box>
        ))}

        {/* Rows */}
        {volSteps.map((v, vIdx) => (
          <React.Fragment key={v}>
            {/* Vol Header */}
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', pr: 2 }}>
              <Typography variant="caption" sx={{ fontWeight: 800, color: v === 0 ? 'primary.main' : 'text.disabled' }}>{v > 0 ? `+${v}` : v}%</Typography>
            </Box>
            {/* Heatmap Cells */}
            {data[vIdx].map((val, pIdx) => (
              <Box 
                key={pIdx} 
                sx={{ 
                  height: 40, 
                  bgcolor: getColor(val), 
                  borderRadius: 0.5, 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  border: val === 0 ? `1px solid ${alpha('#fff', 0.1)}` : 'none',
                  transition: 'all 0.2s ease',
                  '&:hover': { outline: `2px solid ${alpha('#fff', 0.4)}`, zIndex: 1 }
                }}
              >
                <Typography variant="caption" sx={{ fontWeight: 900, color: '#fff', fontSize: '0.65rem', textShadow: '0 1px 2px rgba(0,0,0,0.5)' }}>
                  {val > 0 ? `+${val.toFixed(1)}` : val.toFixed(1)}k
                </Typography>
              </Box>
            ))}
          </React.Fragment>
        ))}
      </Box>
    </Box>
  );
};
