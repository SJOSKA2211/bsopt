import React from 'react';
import { Box, Typography, Button, TextField, Stack, alpha } from '@mui/material';
import { stitchTokens } from '../../../theme/stitch-tokens';

export const OrderTicket: React.FC<{ symbol: string }> = ({ symbol }) => {
  return (
    <Box className="stitch-card" sx={{ p: 0 }}>
      <Box className="stitch-slanted-header">Order Ticket // {symbol}</Box>
      <Box sx={{ p: 2 }}>
        <Stack spacing={2}>
           <Box>
            <Typography className="stitch-label" sx={{ mb: 1 }}>Order Type</Typography>
            <Stack direction="row" spacing={1}>
              {['Market', 'Limit', 'Stop'].map(type => (
                <Button 
                  key={type}
                  sx={{ 
                    flex: 1, 
                    borderRadius: 0, 
                    fontSize: '10px', 
                    fontWeight: 800,
                    bgcolor: type === 'Limit' ? alpha(stitchTokens.colors.primary, 0.1) : 'transparent',
                    border: `1px solid ${type === 'Limit' ? stitchTokens.colors.primary : 'rgba(255,255,255,0.1)'}`,
                    color: type === 'Limit' ? stitchTokens.colors.primary : '#a9abb1'
                  }}
                >
                  {type}
                </Button>
              ))}
            </Stack>
           </Box>

           <Box>
              <Typography className="stitch-label" sx={{ mb: 1 }}>Quantity</Typography>
              <TextField 
                fullWidth 
                variant="standard" 
                defaultValue="100"
                InputProps={{ 
                  className: "stitch-mono",
                  sx: { color: 'white', borderBottom: `1px solid ${alpha(stitchTokens.colors.primary, 0.5)}` },
                  disableUnderline: true 
                }}
              />
           </Box>

           <Box>
              <Typography className="stitch-label" sx={{ mb: 1 }}>Limit Price</Typography>
              <TextField 
                fullWidth 
                variant="standard" 
                defaultValue="189.45"
                InputProps={{ 
                  className: "stitch-mono",
                  sx: { color: 'white', borderBottom: `1px solid ${alpha(stitchTokens.colors.primary, 0.5)}` },
                  disableUnderline: true 
                }}
              />
           </Box>

           <Stack direction="row" spacing={2} sx={{ mt: 2 }}>
              <Button 
                variant="contained" 
                fullWidth 
                sx={{ 
                  borderRadius: 0, 
                  bgcolor: stitchTokens.colors.primary, 
                  color: 'black', 
                  fontWeight: 900,
                  '&:hover': { bgcolor: alpha(stitchTokens.colors.primary, 0.8) }
                }}
              >
                BUY / LONG
              </Button>
              <Button 
                variant="contained" 
                fullWidth 
                sx={{ 
                  borderRadius: 0, 
                  bgcolor: '#ff4d4d', 
                  color: 'white', 
                  fontWeight: 900,
                  '&:hover': { bgcolor: alpha('#ff4d4d', 0.8) }
                }}
              >
                SELL / SHORT
              </Button>
           </Stack>
        </Stack>
      </Box>
    </Box>
  );
};
