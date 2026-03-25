import React from 'react';
import { Box, Typography, Stack, alpha } from '@mui/material';
import { stitchTokens } from '../../../theme/stitch-tokens';

interface DOMRow {
  price: number;
  size: number;
  type: 'bid' | 'ask';
}

export const DOMLadder: React.FC<{ symbol: string; currentPrice: number }> = ({ symbol, currentPrice }) => {
  // Synthetic data for DOM (would be hooked to WS)
  const rows: DOMRow[] = useMemo(() => {
    const arr: DOMRow[] = [];
    for (let i = 10; i > 0; i--) {
      arr.push({ price: currentPrice + i * 0.05, size: Math.floor(Math.random() * 500) + 100, type: 'ask' });
    }
    for (let i = 1; i <= 10; i++) {
        arr.push({ price: currentPrice - i * 0.05, size: Math.floor(Math.random() * 500) + 100, type: 'bid' });
    }
    return arr;
  }, [currentPrice]);

  return (
    <Box className="stitch-card" sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box className="stitch-slanted-header">DOM // {symbol}</Box>
      <Box sx={{ p: 1, flexGrow: 1, overflow: 'auto' }}>
        <Stack spacing={0.2}>
          <Box sx={{ display: 'flex', px: 1, mb: 1 }}>
            <Typography className="stitch-label" sx={{ flex: 1 }}>Size</Typography>
            <Typography className="stitch-label" sx={{ flex: 1, textAlign: 'center' }}>Price</Typography>
            <Typography className="stitch-label" sx={{ flex: 1, textAlign: 'right' }}>Size</Typography>
          </Box>
          {rows.map((row, i) => (
            <Box 
              key={i} 
              sx={{ 
                display: 'flex', 
                height: 24, 
                alignItems: 'center',
                bgcolor: row.price === currentPrice ? alpha(stitchTokens.colors.primary, 0.1) : 'transparent',
                border: row.price === currentPrice ? `1px solid ${alpha(stitchTokens.colors.primary, 0.3)}` : 'none',
              }}
            >
              <Box sx={{ flex: 1, position: 'relative', height: '100%' }}>
                {row.type === 'bid' && (
                  <Box sx={{ 
                    position: 'absolute', 
                    right: 0, 
                    height: '100%', 
                    width: `${Math.min(100, row.size / 10)}%`, 
                    bgcolor: alpha(stitchTokens.colors.primary, 0.15) 
                  }} />
                )}
                <Typography className="stitch-mono" sx={{ position: 'relative', fontSize: '11px', textAlign: 'left', pl: 1, color: row.type === 'bid' ? stitchTokens.colors.primary : 'transparent' }}>
                  {row.type === 'bid' ? row.size : ''}
                </Typography>
              </Box>
              
              <Typography className="stitch-mono" sx={{ flex: 1, textAlign: 'center', fontSize: '11px', fontWeight: 800 }}>
                {row.price.toFixed(2)}
              </Typography>

              <Box sx={{ flex: 1, position: 'relative', height: '100%' }}>
                {row.type === 'ask' && (
                  <Box sx={{ 
                    position: 'absolute', 
                    left: 0, 
                    height: '100%', 
                    width: `${Math.min(100, row.size / 10)}%`, 
                    bgcolor: alpha('#ff4d4d', 0.15) 
                  }} />
                )}
                <Typography className="stitch-mono" sx={{ position: 'relative', fontSize: '11px', textAlign: 'right', pr: 1, color: row.type === 'ask' ? '#ff4d4d' : 'transparent' }}>
                  {row.type === 'ask' ? row.size : ''}
                </Typography>
              </Box>
            </Box>
          ))}
        </Stack>
      </Box>
    </Box>
  );
};

import { useMemo } from 'react';
