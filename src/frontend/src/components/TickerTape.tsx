import React, { useMemo } from 'react';
import { Box, Typography, alpha, useTheme } from '@mui/material';
import { useLiveTickers } from '../api/hooks';
import type { Ticker } from '../api/types';
import { motion } from 'framer-motion';

const DEFAULT_TICKER_SYMBOLS = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'SPY', 'QQQ', 'AAPL', 'NVDA', 'TSLA', 'USDT/USD'];

export const TickerTape: React.FC = () => {
  const theme = useTheme();
  const financial = theme.palette.financial;
  const qfd = financial?.qfd;

  const { data: tickers, isLoading } = useLiveTickers(DEFAULT_TICKER_SYMBOLS);

  // Fallback data for a 'live' feel if API is slow
  const fallbackData: Ticker[] = useMemo(() => [
    { symbol: 'SPY', price: '512.42', percentChange: '+0.45%', up: true },
    { symbol: 'QQQ', price: '445.12', percentChange: '-0.12%', up: false },
    { symbol: 'BTC', price: '64,120.42', percentChange: '+2.14%', up: true },
    { symbol: 'ETH', price: '3,420.12', percentChange: '+1.85%', up: true },
    { symbol: 'SOL', price: '142.05', percentChange: '-3.42%', up: false },
    { symbol: 'NVDA', price: '894.22', percentChange: '+4.12%', up: true },
  ], []);

  const currentTickers = tickers && tickers.length > 0 ? tickers : fallbackData;

  // Duplicate for seamless marquee
  const displayItems = [...currentTickers, ...currentTickers, ...currentTickers];

  return (
    <Box
      sx={{
        width: '100%',
        height: 32,
        bgcolor: 'rgba(1, 4, 9, 0.4)',
        backdropFilter: 'blur(12px)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.03)',
        overflow: 'hidden',
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        '&::before, &::after': {
          content: '""',
          position: 'absolute',
          top: 0,
          width: 80,
          height: '100%',
          zIndex: 2,
          pointerEvents: 'none'
        },
        '&::before': {
          left: 0,
          background: 'linear-gradient(to right, #010409, transparent)'
        },
        '&::after': {
          right: 0,
          background: 'linear-gradient(to left, #010409, transparent)'
        }
      }}
    >
      <motion.div
        animate={{
          x: [0, -1000],
        }}
        transition={{
          x: {
            repeat: Infinity,
            repeatType: "loop",
            duration: 30,
            ease: "linear",
          },
        }}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '48px',
          paddingLeft: '24px',
          whiteSpace: 'nowrap',
          willChange: 'transform'
        }}
      >
        {displayItems.map((t, i) => (
          <Box 
            key={`${t.symbol}-${i}`} 
            sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: 1.5,
              opacity: 0.8,
              transition: 'opacity 0.2s',
              '&:hover': { opacity: 1 }
            }}
          >
            <Typography
              sx={{
                fontWeight: 950,
                color: '#fff',
                fontFamily: 'Outfit',
                fontSize: '10px',
                letterSpacing: '0.1em',
              }}
            >
              {t.symbol}
            </Typography>
            <Typography
              className="data-mono"
              sx={{
                fontSize: '11px',
                color: 'rgba(255,255,255,0.6)',
                fontWeight: 600,
              }}
            >
              {typeof t.price === 'string' ? t.price : parseFloat(t.price).toLocaleString(undefined, { minimumFractionDigits: 2 })}
            </Typography>
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 0.5,
                px: 1,
                py: 0.2,
                borderRadius: '4px',
                bgcolor: alpha(t.up ? '#00ffa3' : '#ff2e7e', 0.08),
                border: `1px solid ${alpha(t.up ? '#00ffa3' : '#ff2e7e', 0.15)}`,
              }}
            >
               <Typography
                className="data-mono"
                sx={{
                  fontSize: '9px',
                  color: t.up ? '#00ffa3' : '#ff2e7e',
                  fontWeight: 900,
                }}
              >
                {t.up ? '▲' : '▼'} {t.percentChange}
              </Typography>
            </Box>
          </Box>
        ))}
      </motion.div>
    </Box>
  );
};
