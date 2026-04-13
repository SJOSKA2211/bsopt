import React from 'react';
import { Box, Typography, Stack, alpha, useTheme, Button } from '@mui/material';

interface PriceLevel {
  price: number;
  bidSize?: number;
  askSize?: number;
  bidBarWidth?: number;
  askBarWidth?: number;
}

const levels: PriceLevel[] = [
  { price: 4.29, askSize: 15, askBarWidth: 20 },
  { price: 4.28, askSize: 18, askBarWidth: 35 },
  { price: 4.27, askSize: 120, askBarWidth: 80 },
  { price: 4.26, askSize: 53, askBarWidth: 45 },
  { price: 4.25, bidSize: 100, askSize: 35, bidBarWidth: 30, askBarWidth: 15 },
  { price: 4.24, bidSize: 200, bidBarWidth: 90 },
  { price: 4.23, bidSize: 150, bidBarWidth: 70 },
  { price: 4.22, bidSize: 50, bidBarWidth: 25 },
  { price: 4.21, bidSize: 25, bidBarWidth: 15 },
  { price: 4.20, bidSize: 100, bidBarWidth: 40 },
];

export const DepthOfMarket: React.FC = () => {
  const theme = useTheme();

  return (
    <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100%', minWidth: 300 }}>
      {/* Header */}
      <Box sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: `1px solid ${alpha('#fff', 0.05)}` }}>
        <Stack direction="row" spacing={1} alignItems="center">
          <Typography variant="caption" sx={{ fontSize: '0.7rem', fontWeight: 800, color: 'text.secondary' }}>DEPTH OF MARKET</Typography>
        </Stack>
      </Box>

      {/* Ladder Column Headers */}
      <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', px: 2, py: 1, borderBottom: `1px solid ${alpha('#fff', 0.05)}` }}>
        <Typography variant="caption" align="center" sx={{ color: 'text.disabled', fontSize: '0.6rem' }}>BID SIZES</Typography>
        <Typography variant="caption" align="center" sx={{ color: 'text.disabled', fontSize: '0.6rem' }}>PRICE</Typography>
        <Typography variant="caption" align="center" sx={{ color: 'text.disabled', fontSize: '0.6rem' }}>ASK SIZE</Typography>
      </Box>

      {/* Ladder Content */}
      <Box sx={{ flexGrow: 1, overflowY: 'auto' }}>
        {levels.map((level, idx) => (
          <Box 
            key={idx} 
            sx={{ 
              display: 'grid', 
              gridTemplateColumns: '1fr 1fr 1fr', 
              height: 40,
              borderBottom: `1px solid ${alpha('#fff', 0.02)}`,
              position: 'relative',
              '&:hover': { bgcolor: alpha('#fff', 0.02) }
            }}
          >
            {/* Bid Side */}
            <Box sx={{ position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'flex-end', px: 2 }}>
              {level.bidSize && (
                <>
                  <Box sx={{ 
                    position: 'absolute', 
                    right: 0, 
                    top: 4, 
                    bottom: 4, 
                    width: `${level.bidBarWidth}%`, 
                    bgcolor: alpha(theme.palette.success.main, 0.4),
                    borderRadius: '4px 0 0 4px',
                  }} />
                  <Typography variant="body2" sx={{ zIndex: 1, fontFamily: 'JetBrains Mono', fontWeight: 700, fontSize: '0.85rem' }}>{level.bidSize}</Typography>
                </>
              )}
            </Box>

            {/* Price Column */}
            <Box sx={{ bgcolor: alpha('#000', 0.2), display: 'flex', alignItems: 'center', justifyContent: 'center', borderLeft: `1px solid ${alpha('#fff', 0.05)}`, borderRight: `1px solid ${alpha('#fff', 0.05)}` }}>
              <Typography variant="body2" sx={{ fontFamily: 'JetBrains Mono', fontWeight: 800, fontSize: '0.9rem', color: level.bidSize && level.askSize ? 'text.primary' : level.bidSize ? 'success.main' : 'error.main' }}>
                {level.price.toFixed(2)}
              </Typography>
            </Box>

            {/* Ask Side */}
            <Box sx={{ position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'flex-start', px: 2 }}>
              {level.askSize && (
                <>
                  <Box sx={{ 
                    position: 'absolute', 
                    left: 0, 
                    top: 4, 
                    bottom: 4, 
                    width: `${level.askBarWidth}%`, 
                    bgcolor: alpha(theme.palette.error.main, 0.4),
                    borderRadius: '0 4px 4px 0'
                  }} />
                  <Typography variant="body2" sx={{ zIndex: 1, fontFamily: 'JetBrains Mono', fontWeight: 700, fontSize: '0.85rem' }}>{level.askSize}</Typography>
                </>
              )}
            </Box>
          </Box>
        ))}
      </Box>

      {/* Execution Buttons Area */}
      <Box sx={{ p: 2, borderTop: `1px solid ${alpha('#fff', 0.05)}`, bgcolor: alpha('#000', 0.2) }}>
        <Stack spacing={1}>
          <Stack direction="row" spacing={1}>
            <Button fullWidth variant="contained" size="small" sx={{ 
              bgcolor: 'success.main', 
              color: '#000', 
              fontWeight: 800,
              '&:hover': { bgcolor: alpha(theme.palette.success.main, 0.8) }
            }}>
              BUY MKT
            </Button>
            <Button fullWidth variant="contained" size="small" sx={{ 
              bgcolor: 'error.main', 
              color: '#fff', 
              fontWeight: 800,
              '&:hover': { bgcolor: alpha(theme.palette.error.main, 0.8) }
            }}>
              SELL MKT
            </Button>
          </Stack>
          <Stack direction="row" spacing={1}>
            <Button fullWidth variant="outlined" size="small" sx={{ 
              borderColor: alpha('#fff', 0.1), 
              color: 'text.primary',
              fontSize: '0.7rem'
            }}>
              Join Bid
            </Button>
            <Button fullWidth variant="outlined" size="small" sx={{ 
              borderColor: alpha('#fff', 0.1), 
              color: 'text.primary',
              fontSize: '0.7rem'
            }}>
              Join Ask
            </Button>
          </Stack>
        </Stack>
      </Box>
    </Box>
  );
};
