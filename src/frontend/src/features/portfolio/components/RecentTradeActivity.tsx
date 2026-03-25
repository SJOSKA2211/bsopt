import React from 'react';
import { Box, Typography, Stack, alpha } from '@mui/material';
import { stitchTokens } from '../../../theme/stitch-tokens';

const trades = [
  { id: 1, type: 'BUY', symbol: 'AAPL', qty: 10, price: 190.50, time: '10:24 AM', status: 'FILLED' },
  { id: 2, type: 'SELL', symbol: 'TSLA', qty: 5, price: 172.15, time: '09:45 AM', status: 'FILLED' },
  { id: 3, type: 'BUY', symbol: 'NVDA', qty: 2, price: 915.00, time: '09:32 AM', status: 'FILLED' },
  { id: 4, type: 'BUY', symbol: 'META', qty: 15, price: 485.20, time: 'Yesterday', status: 'FILLED' },
];

export const RecentTradeActivity: React.FC = () => {
  return (
    <Box>
      <Stack spacing={0.5}>
        {trades.map(trade => (
          <Box 
            key={trade.id} 
            sx={{ 
              p: '10px 16px', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'space-between',
              borderBottom: '1px solid rgba(255,255,255,0.03)',
              '&:hover': { bgcolor: 'rgba(255,255,255,0.01)' }
            }}
          >
            <Stack direction="row" spacing={2} alignItems="center">
              <Box sx={{ 
                width: 4, 
                height: 24, 
                bgcolor: trade.type === 'BUY' ? stitchTokens.colors.primary : '#ff4d4d' 
              }} />
              <Box>
                <Typography sx={{ fontWeight: 800, fontSize: '11px' }}>
                  {trade.type} {trade.qty} {trade.symbol}
                </Typography>
                <Typography className="stitch-mono" sx={{ fontSize: '9px', opacity: 0.6 }}>
                  @ ${trade.price.toFixed(2)}
                </Typography>
              </Box>
            </Stack>
            <Box sx={{ textAlign: 'right' }}>
              <Typography className="stitch-label" sx={{ fontSize: '8px', color: stitchTokens.colors.primary }}>{trade.status}</Typography>
              <Typography className="stitch-mono" sx={{ fontSize: '9px', opacity: 0.4 }}>{trade.time}</Typography>
            </Box>
          </Box>
        ))}
      </Stack>
    </Box>
  );
};
