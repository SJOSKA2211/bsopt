import React from 'react';
import { Box, Typography, Stack, alpha } from '@mui/material';

const sales = [
  { time: '14:32:01', price: 4.25, size: 19, type: 'bid' },
  { time: '14:32:00', price: 4.25, size: 5, type: 'ask' },
  { time: '14:31:58', price: 4.24, size: 100, type: 'ask' },
  { time: '14:31:55', price: 4.24, size: 50, type: 'ask' },
  { time: '14:31:52', price: 4.25, size: 20, type: 'bid' },
  { time: '14:31:48', price: 4.26, size: 12, type: 'bid' },
];

export const TimeAndSales: React.FC = () => {
  return (
    <Box sx={{ 
      height: 40, 
      bgcolor: alpha('#000', 0.4), 
      borderTop: `1px solid ${alpha('#fff', 0.05)}`,
      display: 'flex',
      alignItems: 'center',
      px: 2,
      overflow: 'hidden'
    }}>
      <Typography variant="caption" sx={{ fontWeight: 800, color: 'text.disabled', mr: 3, letterSpacing: '0.1em' }}>TIME & SALES</Typography>
      <Stack direction="row" spacing={4} sx={{ flexGrow: 1 }}>
        {sales.map((sale, idx) => (
          <Stack key={idx} direction="row" spacing={1} alignItems="center">
            <Typography variant="caption" sx={{ color: 'text.disabled', fontSize: '0.65rem' }}>{sale.time}</Typography>
            <Typography variant="body2" sx={{ fontWeight: 800, fontSize: '0.8rem', fontFamily: 'JetBrains Mono' }}>{sale.price.toFixed(2)}</Typography>
            <Typography variant="caption" sx={{ 
              fontWeight: 800, 
              color: sale.type === 'bid' ? 'success.main' : 'error.main',
              bgcolor: alpha(sale.type === 'bid' ? '#00ffa3' : '#ff2e7e', 0.1),
              px: 0.5,
              borderRadius: 0.5,
              fontSize: '0.65rem'
            }}>
              @{sale.size}
            </Typography>
          </Stack>
        ))}
      </Stack>
    </Box>
  );
};
