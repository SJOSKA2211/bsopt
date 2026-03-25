import React from 'react';
import { Box, Typography, Stack, Button, TextField, MenuItem, alpha, useTheme, IconButton } from '@mui/material';
import { Add as AddIcon, Remove as RemoveIcon } from '@mui/icons-material';

export const OrderTicket: React.FC = () => {
  const theme = useTheme();

  return (
    <Box sx={{ p: 3, display: 'flex', flexDirection: 'column', gap: 3, minWidth: 320 }}>
      {/* Header Info */}
      <Stack direction="row" justifyContent="space-between" alignItems="center">
        <Typography variant="h6" sx={{ fontWeight: 800, display: 'flex', alignItems: 'center', gap: 1 }}>
          Order Ticket
        </Typography>
        <Stack direction="row" spacing={1}>
          <Chip label="LMT" size="small" sx={{ bgcolor: 'primary.main', color: '#000', fontWeight: 800, height: 20 }} />
          <Typography variant="caption" sx={{ color: 'text.disabled', fontWeight: 800 }}>MKT</Typography>
          <Typography variant="caption" sx={{ color: 'text.disabled', fontWeight: 800 }}>STP</Typography>
        </Stack>
      </Stack>

      {/* Account & TIF */}
      <Stack direction="row" spacing={2}>
        <Box sx={{ flex: 1 }}>
          <Typography variant="caption" sx={{ color: 'text.disabled', mb: 0.5, display: 'block' }}>ACCOUNT</Typography>
          <TextField
            select
            fullWidth
            size="small"
            value="main"
            sx={{
              '& .MuiOutlinedInput-root': {
                bgcolor: alpha('#fff', 0.03),
                borderRadius: 2,
              }
            }}
          >
            <MenuItem value="main">Main (8392)</MenuItem>
          </TextField>
        </Box>
        <Box sx={{ width: 80 }}>
          <Typography variant="caption" sx={{ color: 'text.disabled', mb: 0.5, display: 'block' }}>TIF</Typography>
          <TextField
            select
            fullWidth
            size="small"
            value="DAY"
            sx={{
              '& .MuiOutlinedInput-root': {
                bgcolor: alpha('#fff', 0.03),
                borderRadius: 2,
              }
            }}
          >
            <MenuItem value="DAY">DAY</MenuItem>
            <MenuItem value="GTC">GTC</MenuItem>
          </TextField>
        </Box>
      </Stack>

      {/* Quantity */}
      <Box>
        <Typography variant="caption" sx={{ color: 'text.disabled', mb: 0.5, display: 'block' }}>QUANTITY</Typography>
        <Stack direction="row" spacing={1} alignItems="center" sx={{ bgcolor: alpha('#fff', 0.03), borderRadius: 2, p: 0.5 }}>
          <IconButton size="small" aria-label="Decrease Quantity"><RemoveIcon fontSize="small" /></IconButton>
          <Typography variant="h5" align="center" sx={{ flexGrow: 1, fontFamily: 'JetBrains Mono', fontWeight: 800 }}>18</Typography>
          <IconButton size="small" aria-label="Increase Quantity"><AddIcon fontSize="small" /></IconButton>
        </Stack>
        <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
          {['10', '50', '100', 'MAX'].map(val => (
            <Button key={val} size="small" variant="outlined" sx={{ 
              minWidth: 0, 
              flex: 1, 
              fontSize: '0.65rem', 
              borderColor: alpha('#fff', 0.1),
              color: 'text.secondary'
            }}>
              {val}
            </Button>
          ))}
        </Stack>
      </Box>

      {/* Limit Price */}
      <Box>
        <Typography variant="caption" sx={{ color: 'text.disabled', mb: 0.5, display: 'block' }}>LIMIT PRICE</Typography>
        <Stack direction="row" spacing={1} alignItems="center" sx={{ bgcolor: alpha('#fff', 0.03), borderRadius: 2, p: 0.5 }}>
          <IconButton size="small" aria-label="Decrease Limit Price"><RemoveIcon fontSize="small" /></IconButton>
          <Typography variant="h5" align="center" sx={{ flexGrow: 1, fontFamily: 'JetBrains Mono', fontWeight: 800 }}>4.25</Typography>
          <IconButton size="small" aria-label="Increase Limit Price"><AddIcon fontSize="small" /></IconButton>
        </Stack>
      </Box>

      {/* Stats */}
      <Stack spacing={0.5} sx={{ borderTop: `1px solid ${alpha('#fff', 0.05)}`, pt: 2 }}>
        <Stack direction="row" justifyContent="space-between">
          <Typography variant="caption" sx={{ color: 'text.disabled' }}>Est. Margin</Typography>
          <Typography variant="body2" sx={{ fontWeight: 800, fontFamily: 'JetBrains Mono' }}>$4,250.00</Typography>
        </Stack>
        <Stack direction="row" justifyContent="space-between">
          <Typography variant="caption" sx={{ color: 'text.disabled' }}>Post-Trade Delta</Typography>
          <Typography variant="body2" sx={{ fontWeight: 800, color: 'success.main', fontFamily: 'JetBrains Mono' }}>+45.20</Typography>
        </Stack>
      </Stack>

      {/* Action Buttons */}
      <Stack spacing={1.5}>
        <Button 
          fullWidth 
          variant="contained" 
          sx={{ 
            height: 54, 
            borderRadius: 2,
            background: 'linear-gradient(135deg, #00ffa3 0%, #00b372 100%)',
            color: '#000',
            fontWeight: 800,
            fontSize: '1rem',
          }}
        >
          PLACE BUY ORDER
        </Button>
        <Button 
          fullWidth 
          variant="contained" 
          sx={{ 
            height: 54, 
            borderRadius: 2,
            background: 'linear-gradient(135deg, #ff2e7e 0%, #c20058 100%)',
            color: '#fff',
            fontWeight: 800,
            fontSize: '1rem',
          }}
        >
          PLACE SELL ORDER
        </Button>
      </Stack>
    </Box>
  );
};

const Chip = ({ label, size, sx }: any) => (
  <Box sx={{ 
    px: 1, 
    borderRadius: 1, 
    display: 'flex', 
    alignItems: 'center', 
    justifyContent: 'center',
    fontSize: '0.65rem',
    ...sx 
  }}>
    {label}
  </Box>
);
