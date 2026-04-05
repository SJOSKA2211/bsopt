import React, { useMemo } from 'react';
import { Box, Typography, Stack, alpha } from '@mui/material';
import { stitchTokens } from '../../../theme/stitch-tokens';
import { usePricingStore } from '../../../store/usePricingStore';
import type { PricingState } from '../../../store/usePricingStore';

interface DOMRow {
  price: number;
  size: number;
  type: 'bid' | 'ask';
}

export const DOMLadder: React.FC<{ symbol: string }> = ({ symbol }) => {
  const priceData = usePricingStore((state: PricingState) => state.prices[symbol]);
  const currentPrice = priceData?.price ?? 189.45;

  // Synthetic data for DOM (would be hooked to WS)
  const rows: DOMRow[] = useMemo(() => {
    const arr: DOMRow[] = [];
    // ASKS (Red)
    for (let i = 12; i > 0; i--) {
      arr.push({ price: currentPrice + i * 0.05, size: Math.floor(Math.random() * 800) + 200, type: 'ask' });
    }
    // BIDS (Green)
    for (let i = 1; i <= 12; i++) {
      arr.push({ price: currentPrice - i * 0.05, size: Math.floor(Math.random() * 800) + 200, type: 'bid' });
    }
    return arr;
  }, [currentPrice]);

  return (
    <Box className="stitch-card" sx={{ height: '100%', display: 'flex', flexDirection: 'column', p: 0, position: 'relative' }}>
      <Box className="stitch-dots-container" sx={{ opacity: 0.03 }} />
      <Box className="stitch-slanted-header" sx={{ bgcolor: '#121418', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
        <Typography sx={{ fontSize: '10px', fontWeight: 900, letterSpacing: '1px' }}>DOM_LADDER // {symbol}</Typography>
      </Box>
      <Box sx={{ p: '8px 4px', flexGrow: 1, overflow: 'auto', position: 'relative' }}>
        <Stack spacing={0.1}>
          <Box sx={{ display: 'flex', px: 1, mb: 1, borderBottom: '1px solid rgba(255,255,255,0.05)', pb: 0.5 }}>
            <Typography className="stitch-label" sx={{ flex: 1, fontSize: '8px' }}>BID_SIZE</Typography>
            <Typography className="stitch-label" sx={{ flex: 1, textAlign: 'center', fontSize: '8px' }}>PRICE_UNIT</Typography>
            <Typography className="stitch-label" sx={{ flex: 1, textAlign: 'right', fontSize: '8px' }}>ASK_SIZE</Typography>
          </Box>
          {rows.map((row, i) => {
            const isCurrent = Math.abs(row.price - currentPrice) < 0.02;
            return (
              <Box 
                key={i} 
                sx={{ 
                  display: 'flex', 
                  height: 20, 
                  alignItems: 'center',
                  position: 'relative',
                  bgcolor: isCurrent ? alpha(stitchTokens.colors.primary, 0.1) : 'transparent',
                  cursor: 'crosshair',
                  '&:hover': { bgcolor: 'rgba(255,255,255,0.03)' }
                }}
              >
                {/* BID BAR */}
                <Box sx={{ flex: 1, position: 'relative', height: '100%' }}>
                  {row.type === 'bid' && (
                    <Box sx={{ 
                      position: 'absolute', 
                      right: 0, 
                      height: '100%', 
                      width: `${Math.min(100, row.size / 10)}%`, 
                      background: `linear-gradient(to left, ${alpha(stitchTokens.colors.primary, 0.2)}, transparent)`,
                      borderRight: `1px solid ${alpha(stitchTokens.colors.primary, 0.4)}`
                    }} />
                  )}
                  <Typography className="stitch-mono" sx={{ position: 'relative', fontSize: '10px', fontWeight: 700, textAlign: 'left', pl: 1, color: row.type === 'bid' ? stitchTokens.colors.primary : 'transparent' }}>
                    {row.type === 'bid' ? row.size : ''}
                  </Typography>
                </Box>
                
                {/* PRICE */}
                <Typography className="stitch-mono" sx={{ 
                  flex: 0.8, 
                  textAlign: 'center', 
                  fontSize: '10px', 
                  fontWeight: 900,
                  color: isCurrent ? stitchTokens.colors.primary : '#fff',
                  textShadow: isCurrent ? `0 0 10px ${stitchTokens.colors.primary}` : 'none'
                }}>
                  {row.price.toFixed(2)}
                </Typography>
  
                {/* ASK BAR */}
                <Box sx={{ flex: 1, position: 'relative', height: '100%' }}>
                  {row.type === 'ask' && (
                    <Box sx={{ 
                      position: 'absolute', 
                      left: 0, 
                      height: '100%', 
                      width: `${Math.min(100, row.size / 10)}%`, 
                      background: `linear-gradient(to right, ${alpha('#ff2e7e', 0.2)}, transparent)`,
                      borderLeft: `1px solid ${alpha('#ff2e7e', 0.4)}`
                    }} />
                  )}
                  <Typography className="stitch-mono" sx={{ position: 'relative', fontSize: '10px', fontWeight: 700, textAlign: 'right', pr: 1, color: row.type === 'ask' ? '#ff2e7e' : 'transparent' }}>
                    {row.type === 'ask' ? row.size : ''}
                  </Typography>
                </Box>
              </Box>
            );
          })}
        </Stack>
      </Box>
      <Box sx={{ p: 1, bgcolor: 'rgba(0,0,0,0.3)', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
         <Stack direction="row" spacing={2} justifyContent="space-between">
            <Typography className="stitch-label" sx={{ fontSize: '7px' }}>VWAP: 189.42</Typography>
            <Typography className="stitch-label" sx={{ fontSize: '7px' }}>VOL: 1.2M</Typography>
         </Stack>
      </Box>
    </Box>
  );
};
