import React, { useMemo } from 'react';
import { Box, Typography, alpha, useTheme } from '@mui/material';
import { useLiveTickers } from '../api/hooks';
import type { Ticker } from '../api/types';
import { motion } from 'framer-motion';

// Default symbols for the ticker tape
const DEFAULT_TICKER_SYMBOLS = ['SPY', 'QQQ', 'AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META'];

export const TickerTape: React.FC = () => {
  const theme = useTheme();
  const financial = theme.palette.financial;
  const qfd = financial?.qfd;

  const { data: tickers, isLoading } = useLiveTickers(DEFAULT_TICKER_SYMBOLS);

  // Create a continuous scrolling loop by duplicating elements
  const displayTickers = useMemo(() => {
    if (!tickers || tickers.length === 0) return [];
    // Triple the tickers to ensure seamless infinite loop on wide screens
    return [...tickers, ...tickers, ...tickers];
  }, [tickers]);

  if (isLoading && !tickers) {
    return (
      <Box
        sx={{
          width: '100%',
          height: 32,
          bgcolor: 'rgba(11, 14, 18, 0.5)',
          backdropFilter: 'blur(20px)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
          display: 'flex',
          alignItems: 'center',
          px: 3,
        }}
      >
        <div className="stitch-live-indicator" style={{ marginRight: 12, width: 6, height: 6 }} />
        <Typography
          variant="caption"
          sx={{ 
            color: 'rgba(255,255,255,0.4)', 
            fontWeight: 900, 
            letterSpacing: '0.2em',
            fontSize: '9px',
            fontFamily: 'Space Grotesk' 
          }}
        >
          INITIALIZING_MARKET_MESH_STREAM...
        </Typography>
      </Box>
    );
  }

  if (displayTickers.length === 0) {
    return null; // Don't show empty bar if no data
  }

  return (
    <Box
      sx={{
        width: '100%',
        height: 32,
        bgcolor: 'rgba(11, 14, 18, 0.6)',
        backdropFilter: 'blur(30px)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.03)',
        overflow: 'hidden',
        display: 'flex',
        alignItems: 'center',
        flexShrink: 0,
        zIndex: 5
      }}
    >
      <Box className="ticker-strip">
        <Box className="ticker-track">
          {displayTickers.map((t: Ticker, i: number) => (
            <Box key={`${t.symbol}-${i}`} className="ticker-item">
              <Stack direction="row" spacing={1.5} alignItems="baseline">
                <Typography
                  sx={{
                    fontWeight: 950,
                    color: '#fff',
                    fontFamily: 'Outfit',
                    fontSize: '10px',
                    letterSpacing: '0.05em',
                  }}
                >
                  {t.symbol}
                </Typography>
                <Typography
                  className="stitch-mono"
                  sx={{
                    fontSize: '10px',
                    color: 'rgba(255,255,255,0.7)',
                    fontWeight: 600,
                  }}
                >
                  {parseFloat(t.price).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </Typography>
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 0.5,
                    px: 0.8,
                    py: 0.1,
                    bgcolor: alpha(t.up ? (qfd?.emerald ?? '#00ffa3') : '#ff2e7e', 0.1),
                    border: `1px solid ${alpha(t.up ? (qfd?.emerald ?? '#00ffa3') : '#ff2e7e', 0.2)}`,
                  }}
                >
                   <Typography
                    className="stitch-mono"
                    sx={{
                      fontSize: '9px',
                      color: t.up ? (qfd?.emerald ?? '#00ffa3') : '#ff2e7e',
                      fontWeight: 900,
                    }}
                  >
                    {t.up ? '▲' : '▼'} {t.percentChange}
                  </Typography>
                </Box>
              </Stack>
            </Box>
          ))}
        </Box>
      </Box>
    </Box>
  );
};

// Internal Stack helper if not imported from MUI
const Stack: React.FC<{ children: React.ReactNode, direction?: 'row' | 'column', spacing?: number, alignItems?: string }> = ({ children, direction = 'row', spacing = 1, alignItems = 'center' }) => (
  <Box sx={{ display: 'flex', flexDirection: direction, gap: spacing * 8 + 'px', alignItems }}>
    {children}
  </Box>
);
