import React, { useMemo } from 'react';
import { Box, Typography, alpha } from '@mui/material';
import { motion } from 'framer-motion';

export const TickerTape: React.FC = () => {
  const displayItems = useMemo(() => [
    { symbol: 'SPY', price: '512.42', percentChange: '+0.45%', up: true },
    { symbol: 'QQQ', price: '445.12', percentChange: '-0.12%', up: false },
    { symbol: 'BTC/USD', price: '64,120.42', percentChange: '+2.14%', up: true },
    { symbol: 'ETH/USD', price: '3,420.12', percentChange: '+1.85%', up: true },
    { symbol: 'NVDA', price: '894.22', percentChange: '+4.12%', up: true },
    { symbol: 'SPY', price: '512.42', percentChange: '+0.45%', up: true },
    { symbol: 'QQQ', price: '445.12', percentChange: '-0.12%', up: false },
    { symbol: 'BTC/USD', price: '64,120.42', percentChange: '+2.14%', up: true },
  ], []);

  return (
    <Box sx={{ width: '100%', height: 32, bgcolor: 'rgba(0,0,0,0.5)', borderBottom: '1px solid rgba(255,255,255,0.03)', overflow: 'hidden', position: 'relative', display: 'flex', alignItems: 'center' }}>
      <motion.div animate={{ x: [0, -1200] }} transition={{ x: { repeat: Infinity, duration: 40, ease: "linear" } }} style={{ display: 'flex', alignItems: 'center', gap: '48px', paddingLeft: '48px', whiteSpace: 'nowrap' }}>
        {displayItems.map((t, i) => (
          <Box key={i} sx={{ display: 'flex', alignItems: 'center', gap: 1.5, opacity: 0.8 }}>
            <Typography sx={{ fontWeight: 950, color: '#fff', fontSize: '9px', letterSpacing: '0.1em' }}>{t.symbol}</Typography>
            <Typography className="data-mono" sx={{ fontSize: '10px', color: 'rgba(255,255,255,0.5)' }}>{t.price}</Typography>
            <Box sx={{ px: 0.8, py: 0.1, borderRadius: '4px', bgcolor: alpha(t.up ? '#00ffa3' : '#ef4444', 0.1), border: `1px solid ${alpha(t.up ? '#00ffa3' : '#ef4444', 0.2)}` }}>
               <Typography sx={{ fontSize: '8px', color: t.up ? '#00ffa3' : '#ef4444', fontWeight: 900 }}>{t.up ? '▲' : '▼'} {t.percentChange}</Typography>
            </Box>
          </Box>
        ))}
      </motion.div>
    </Box>
  );
};
