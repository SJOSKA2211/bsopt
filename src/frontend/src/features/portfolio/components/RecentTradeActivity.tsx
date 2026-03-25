import React from 'react';
import { Box, Typography, Stack, alpha } from '@mui/material';
import { stitchTokens } from '../../../theme/stitch-tokens';

const trades = [
  { id: 1, type: 'BUY_LONG', symbol: 'AAPL_EQUITY', qty: '10.0', price: 190.50, time: '10:24:12', status: 'FILLED_TOTAL' },
  { id: 2, type: 'SELL_SHORT', symbol: 'TSLA_EQUITY', qty: '5.0', price: 172.15, time: '09:45:05', status: 'FILLED_TOTAL' },
  { id: 3, type: 'BUY_LONG', symbol: 'NVDA_EQUITY', qty: '2.0', price: 915.00, time: '09:32:18', status: 'FILLED_PART' },
  { id: 4, type: 'BUY_LONG', symbol: 'META_EQUITY', qty: '15.0', price: 485.20, time: '08:12:44', status: 'FILLED_TOTAL' },
];

export const RecentTradeActivity: React.FC = () => {
  return (
    <Box sx={{ position: 'relative', overflow: 'hidden' }}>
      <Box className="stitch-dots-container" sx={{ opacity: 0.05 }} />
      <Stack spacing={0.1}>
        {trades.map((trade, idx) => (
          <Box 
            key={trade.id} 
            sx={{ 
              p: '12px 16px', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'space-between',
              borderBottom: '1px solid rgba(255,255,255,0.03)',
              bgcolor: idx % 2 === 0 ? 'rgba(255,255,255,0.01)' : 'transparent',
              '&:hover': { bgcolor: 'rgba(255,255,255,0.03)' },
              position: 'relative'
            }}
          >
            <Stack direction="row" spacing={2} alignItems="center">
              <Box sx={{ 
                width: 2, 
                height: 28, 
                bgcolor: trade.type.includes('BUY') ? stitchTokens.colors.primary : '#ff2e7e',
                boxShadow: `0 0 10px ${alpha(trade.type.includes('BUY') ? stitchTokens.colors.primary : '#ff2e7e', 0.4)}`
              }} />
              <Box>
                <Typography sx={{ fontWeight: 950, fontSize: '10px', color: '#fff', letterSpacing: '0.5px' }}>
                  {trade.type} // {trade.symbol}
                </Typography>
                <Typography className="stitch-mono" sx={{ fontSize: '9px', opacity: 0.5, fontWeight: 700 }}>
                   Qty: {trade.qty} @ <Box component="span" sx={{ color: stitchTokens.colors.primary }}>${trade.price.toFixed(2)}</Box>
                </Typography>
              </Box>
            </Stack>
            <Box sx={{ textAlign: 'right' }}>
              <Typography className="stitch-label" sx={{ fontSize: '8px', color: stitchTokens.colors.primary, fontWeight: 900, mb: 0.5 }}>{trade.status}</Typography>
              <Typography className="stitch-mono" sx={{ fontSize: '9px', opacity: 0.3, fontWeight: 900 }}>{trade.time}</Typography>
            </Box>
            
            {/* Abstract Geometric Decoration */}
            {idx === 0 && (
              <Box className="stitch-abstract-shard" sx={{ position: 'absolute', bottom: -10, right: 40, width: 60, height: 60, bgcolor: 'rgba(0,255,163,0.02)', clipPath: stitchTokens.geometry.shard }} />
            )}
          </Box>
        ))}
      </Stack>
      <Box sx={{ p: '6px 12px', bgcolor: 'rgba(0,0,0,0.3)', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
         <Typography className="stitch-label" sx={{ fontSize: '7px', textAlign: 'center', opacity: 0.4 }}>VIEW_ALL_HISTORICAL_TRANSACTIONS // ARCHIVE_v8.4</Typography>
      </Box>
    </Box>
  );
};
