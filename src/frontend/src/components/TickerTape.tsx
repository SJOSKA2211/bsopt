import React from 'react';
import { Box, Typography, alpha, useTheme } from '@mui/material';
import { useMarketTickers } from '../api/hooks';
import type { Ticker } from '../api/types';

export const TickerTape: React.FC = () => {
  const theme = useTheme();
  const financial = theme.palette.financial;
  const qfd = financial?.qfd;

  const { data: tickers, isLoading } = useMarketTickers();

  // Create a continuous scrolling loop by duplicating elements
  const displayTickers = tickers && tickers.length > 0 ? [...tickers, ...tickers] : [];

  if (isLoading && !tickers) {
    return (
      <Box
        sx={{
          width: '100%',
          height: 40,
          bgcolor: alpha(theme.palette.background.paper, 0.4),
          backdropFilter: 'blur(20px)',
          borderBottom: `1px solid ${alpha('#fff', 0.05)}`,
          display: 'flex',
          alignItems: 'center',
          px: 3,
        }}
      >
        <Typography
          variant="caption"
          sx={{ color: 'text.secondary', fontWeight: 800, letterSpacing: '0.1em' }}
        >
          SYNCHRONIZING GLOBAL TAPE...
        </Typography>
      </Box>
    );
  }

  if (displayTickers.length === 0) {
    return (
      <Box
        sx={{
          width: '100%',
          height: 40,
          bgcolor: alpha(theme.palette.background.paper, 0.4),
          backdropFilter: 'blur(20px)',
          borderBottom: `1px solid ${alpha('#fff', 0.05)}`,
          display: 'flex',
          alignItems: 'center',
          px: 3,
        }}
      >
        <Typography
          variant="caption"
          sx={{ color: 'text.secondary', fontWeight: 800, letterSpacing: '0.1em' }}
        >
          AWAITING MARKET DATA...
        </Typography>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        width: '100%',
        height: 40,
        bgcolor: alpha(theme.palette.background.paper, 0.4),
        backdropFilter: 'blur(30px)',
        borderBottom: `1px solid ${alpha('#fff', 0.03)}`,
        overflow: 'hidden',
        display: 'flex',
        alignItems: 'center',
        flexShrink: 0,
      }}
    >
      <Box className="ticker-strip">
        <Box className="ticker-track">
          {displayTickers.map((t: Ticker, i: number) => (
            <Box key={`${t.symbol}-${i}`} className="ticker-item">
              <Typography
                variant="caption"
                sx={{
                  fontWeight: 900,
                  color: 'text.primary',
                  fontFamily: 'Outfit',
                  fontSize: '0.75rem',
                  letterSpacing: '0.02em',
                }}
              >
                {t.symbol}
              </Typography>
              <Typography
                variant="caption"
                sx={{
                  fontFamily: 'JetBrains Mono',
                  fontSize: '0.7rem',
                  color: 'text.secondary',
                  fontWeight: 600,
                }}
              >
                ${parseFloat(t.price).toFixed(2)}
              </Typography>
              <Typography
                variant="caption"
                sx={{
                  fontFamily: 'JetBrains Mono',
                  fontSize: '0.65rem',
                  color: t.up ? qfd?.emerald ?? '#10b981' : theme.palette.error.main,
                  fontWeight: 900,
                }}
              >
                {t.percentChange}
              </Typography>
            </Box>
          ))}
        </Box>
      </Box>
    </Box>
  );
};
