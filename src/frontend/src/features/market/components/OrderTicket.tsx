import React from 'react';
import { Box, Typography, Button, TextField, Stack, alpha } from '@mui/material';
import { stitchTokens } from '../../../theme/stitch-tokens';

export const OrderTicket: React.FC<{ symbol: string }> = ({ symbol }) => {
  return (
    <Box className="stitch-card" sx={{ p: 0, position: 'relative', overflow: 'hidden' }}>
      <Box className="stitch-dots-container" sx={{ opacity: 0.03 }} />
      <Box className="stitch-slanted-header" sx={{ bgcolor: stitchTokens.colors.primary, color: 'black', borderBottom: 'none' }}>
        <Typography sx={{ fontSize: '10px', fontWeight: 950, color: 'black', letterSpacing: '1px' }}>EXECUTION_ENGINE // {symbol}</Typography>
      </Box>
      <Box sx={{ p: 2 }}>
        <Stack spacing={2.5}>
          <Box>
            <Typography className="stitch-label" sx={{ mb: 1, fontSize: '8px', opacity: 0.6 }}>ORDER_MODE_SELECT</Typography>
            <Stack direction="row" spacing={0.5}>
              {['MARKET', 'LIMIT', 'STOP', 'O-C-O'].map(type => (
                <Button
                  key={type}
                  sx={{
                    flex: 1,
                    borderRadius: 0,
                    height: 28,
                    fontSize: '9px',
                    fontWeight: 900,
                    bgcolor: type === 'LIMIT' ? alpha(stitchTokens.colors.primary, 0.1) : 'rgba(255,255,255,0.02)',
                    border: `1px solid ${type === 'LIMIT' ? stitchTokens.colors.primary : 'rgba(255,255,255,0.05)'}`,
                    color: type === 'LIMIT' ? stitchTokens.colors.primary : 'rgba(255,255,255,0.4)',
                    '&:hover': { bgcolor: 'rgba(255,255,255,0.05)' }
                  }}
                >
                  {type}
                </Button>
              ))}
            </Stack>
          </Box>

          <Box>
            <Stack direction="row" spacing={2}>
              <Box sx={{ flex: 1 }}>
                <Typography className="stitch-label" sx={{ mb: 1, fontSize: '8px', opacity: 0.6 }}>QUANTITY_UNITS</Typography>
                <Box sx={{ p: 1, bgcolor: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <Typography className="stitch-mono" sx={{ fontSize: '12px', fontWeight: 900 }}>100</Typography>
                </Box>
              </Box>
              <Box sx={{ flex: 1 }}>
                <Typography className="stitch-label" sx={{ mb: 1, fontSize: '8px', opacity: 0.6 }}>LIMIT_PRICE_USD</Typography>
                <Box sx={{ p: 1, bgcolor: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <Typography className="stitch-mono" sx={{ fontSize: '12px', fontWeight: 900, color: stitchTokens.colors.primary }}>189.45</Typography>
                </Box>
              </Box>
            </Stack>
          </Box>

          <Box>
            <Typography className="stitch-label" sx={{ mb: 1, fontSize: '8px', opacity: 0.6 }}>DURATION_PARAMS</Typography>
            <Box className="stitch-banner-orange" style={{ fontSize: '8px', width: 'fit-content', padding: '2px 10px' }}>DAY_ONLY // GTC_DISABLED</Box>
          </Box>

          <Stack direction="row" spacing={1.5} sx={{ mt: 1 }}>
            <Button
              variant="contained"
              fullWidth
              sx={{
                borderRadius: 0,
                height: 42,
                bgcolor: stitchTokens.colors.primary,
                color: 'black',
                fontSize: '11px',
                fontWeight: 950,
                letterSpacing: '1px',
                boxShadow: `0 0 20px ${alpha(stitchTokens.colors.primary, 0.3)}`,
                '&:hover': { bgcolor: alpha(stitchTokens.colors.primary, 0.8) }
              }}
            >
              BUY_EXECUTE_L
            </Button>
            <Button
              variant="contained"
              fullWidth
              sx={{
                borderRadius: 0,
                height: 42,
                bgcolor: '#ff2e7e',
                color: 'white',
                fontSize: '11px',
                fontWeight: 950,
                letterSpacing: '1px',
                boxShadow: `0 0 20px ${alpha('#ff2e7e', 0.3)}`,
                '&:hover': { bgcolor: alpha('#ff2e7e', 0.8) }
              }}
            >
              SELL_EXECUTE_S
            </Button>
          </Stack>
        </Stack>
      </Box>
      <Box sx={{ p: '6px 12px', bgcolor: 'rgba(0,0,0,0.3)', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
        <Typography className="stitch-label" sx={{ fontSize: '7px', opacity: 0.4 }}>COMMISSION_EST: $0.45 // EXCH_FEE: $0.02</Typography>
      </Box>
    </Box>
  );
};
