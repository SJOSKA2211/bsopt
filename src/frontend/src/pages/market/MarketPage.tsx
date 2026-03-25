import React, { lazy, Suspense, useState, useCallback } from 'react';
import {
  Box,
  Typography,
  CircularProgress,
  Stack,
  alpha,
  Grid,
} from '@mui/material';
import { usePricingStore } from '../../store/usePricingStore';
import type { PricingState } from '../../store/usePricingStore';
import { useDataIntegration } from '../../hooks/useDataIntegration';
import { stitchTokens } from '../../theme/stitch-tokens';
import { DOMLadder } from '../../features/market/components/DOMLadder';
import { LevelIIQuotes } from '../../features/market/components/LevelIIQuotes';
import { OrderTicket } from '../../features/market/components/OrderTicket';

// Lazy loaded trading components
const LivePriceChart = lazy(() =>
  import('../../features/charts/components/LivePriceChart').then(m => ({ default: m.LivePriceChart }))
);
const OptionsChain = lazy(() =>
  import('../../features/options/components/OptionsChain').then(m => ({ default: m.OptionsChain }))
);

const LoadingFallback: React.FC = () => (
  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', minHeight: 300 }}>
    <CircularProgress size={24} sx={{ color: stitchTokens.colors.primary }} />
  </Box>
);

export const MarketPage: React.FC = () => {
  const [currentSymbol] = useState('AAPL');
  
  // Establish unified real-time connection
  const { isConnected } = useDataIntegration({ symbols: [currentSymbol] });
  
  // Get live price from store
  const priceData = usePricingStore((state: PricingState) => state.prices[currentSymbol]);
  const livePrice = priceData?.price ?? 189.45;

  return (
    <Box sx={{ p: 2, height: 'calc(100vh - 72px)', overflow: 'hidden' }}>
      <Grid container spacing={2} sx={{ height: '100%' }}>
        {/* Left Column: Analytics & Chain */}
        <Grid item xs={12} lg={8.5} sx={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 2 }}>
          {/* Main Chart Card */}
          <Box className="stitch-card" sx={{ flex: 1, minHeight: 0 }}>
             <Box className="stitch-slanted-header">Live Chart // {currentSymbol}</Box>
             <Box sx={{ p: 1, height: 'calc(100% - 32px)' }}>
                <Suspense fallback={<LoadingFallback />}>
                  <LivePriceChart symbol={currentSymbol} />
                </Suspense>
             </Box>
          </Box>

          {/* Options Chain Card */}
          <Box className="stitch-card" sx={{ flex: 1, minHeight: 0 }}>
             <Box className="stitch-slanted-header" sx={{ bgcolor: stitchTokens.colors.tertiary }}>Option Chain // Dec 2024</Box>
             <Box sx={{ height: 'calc(100% - 32px)', overflow: 'auto' }}>
                <Suspense fallback={<LoadingFallback />}>
                  <OptionsChain symbol={currentSymbol} />
                </Suspense>
             </Box>
          </Box>
        </Grid>

        {/* Right Column: execution & depth */}
        <Grid item xs={12} lg={3.5} sx={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 2 }}>
          {/* DOM Ladder */}
          <Box sx={{ flex: 1.5, minHeight: 0 }}>
             <DOMLadder symbol={currentSymbol} currentPrice={livePrice} />
          </Box>

          {/* Level II Quotes */}
          <Box sx={{ flex: 1, minHeight: 0 }}>
             <LevelIIQuotes symbol={currentSymbol} />
          </Box>

          {/* Order Ticket */}
          <Box sx={{ flex: 1, minHeight: 0 }}>
             <OrderTicket symbol={currentSymbol} />
          </Box>
        </Grid>
      </Grid>

      {/* Connection Toast (Overlay) */}
      {!isConnected && (
        <Box sx={{ 
          position: 'fixed', 
          bottom: 24, 
          right: 24, 
          className: "stitch-card", 
          p: 2, 
          bgcolor: 'rgba(255, 107, 107, 0.2)',
          border: '1px solid #ff6b6b'
        }}>
          <Typography className="stitch-label" sx={{ color: '#ff6b6b' }}>
            Data Pipeline: Reconnecting...
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default MarketPage;
