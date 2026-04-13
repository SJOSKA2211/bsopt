import React from 'react';
import { Box, Typography, alpha, useTheme } from '@mui/material';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

interface RiskHeatmapProps {
  spot?: number;
  strike?: number;
  timeToExpiry?: number;
  volatility?: number;
  rate?: number;
  optionType?: 'call' | 'put';
}

const priceSteps = [-10, -5, -2, 0, 2, 5, 10];
const volSteps = [-5, -2, 0, 2, 5];

export const RiskHeatmap: React.FC<RiskHeatmapProps> = ({
  spot = 100,
  strike = 100,
  timeToExpiry = 1.0,
  volatility = 0.2,
  rate = 0.05,
  optionType = 'call'
}) => {
  const theme = useTheme();

  const { data: heatmapResponse, isLoading } = useQuery({
    queryKey: ['heatmap', spot, strike, timeToExpiry, volatility, rate, optionType],
    queryFn: async () => {
      const resp = await axios.post('/api/v1/pricing/heatmap', {
        spot,
        strike,
        time_to_expiry: timeToExpiry,
        volatility,
        rate,
        option_type: optionType,
        price_shifts: priceSteps,
        vol_shifts: volSteps
      });
      return resp.data;
    },
    staleTime: 30000,
  });

  const getColor = (val: number) => {
    if (val > 0) return alpha(theme.palette.success.main, Math.min(Math.abs(val) / 5.0, 0.8));
    if (val < 0) return alpha(theme.palette.error.main, Math.min(Math.abs(val) / 5.0, 0.8));
    return alpha('#fff', 0.05);
  };

  if (isLoading || !heatmapResponse) {
    return (
      <Box sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="caption" sx={{ color: 'text.disabled' }}>CALCULATING RISK SURFACE...</Typography>
      </Box>
    );
  }

  const grid = heatmapResponse.grid;

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="caption" sx={{ color: 'text.disabled', mb: 3, display: 'block', textAlign: 'center' }}>
        P&L IMPACT SURFACE: PRICE (%) vs VOLATILITY (pts)
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
        {grid.map((row: any, vIdx: number) => {
          const v = volSteps[vIdx];
          return (
            <React.Fragment key={v}>
              {/* Vol Header */}
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', pr: 2 }}>
                <Typography variant="caption" sx={{ fontWeight: 800, color: v === 0 ? 'primary.main' : 'text.disabled' }}>{v > 0 ? `+${v}` : v}%</Typography>
              </Box>
              {/* Heatmap Cells */}
              {row.map((cell: any, pIdx: number) => (
                <Box 
                  key={pIdx} 
                  sx={{ 
                    height: 40, 
                    bgcolor: getColor(cell.pnl), 
                    borderRadius: 0.5, 
                    display: 'flex', 
                    flexDirection: 'column',
                    alignItems: 'center', 
                    justifyContent: 'center',
                    border: cell.pnl === 0 ? `1px solid ${alpha('#fff', 0.1)}` : 'none',
                    transition: 'all 0.2s ease',
                    '&:hover': { 
                        outline: `2px solid ${alpha('#fff', 0.4)}`, 
                        zIndex: 1,
                        transform: 'scale(1.05)'
                    }
                  }}
                >
                  <Typography variant="caption" sx={{ fontWeight: 900, color: '#fff', fontSize: '0.65rem', textShadow: '0 1px 2px rgba(0,0,0,0.5)' }}>
                    {cell.pnl > 0 ? `+${cell.pnl.toFixed(2)}` : cell.pnl.toFixed(2)}
                  </Typography>
                </Box>
              ))}
            </React.Fragment>
          );
        })}
      </Box>
    </Box>
  );
};
