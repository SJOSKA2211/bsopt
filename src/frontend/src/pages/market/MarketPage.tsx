import React, { lazy, Suspense, useState } from 'react';
import {
  Box,
  Typography,
  CircularProgress,
  Grid,
} from '@mui/material';
import { AnimatePresence, motion } from 'framer-motion';
import { useDataIntegration } from '../../hooks/useDataIntegration';
import { stitchTokens } from '../../theme/stitch-tokens';
import { DOMLadder } from '../../features/market/components/DOMLadder';
import { LevelIIQuotes } from '../../features/market/components/LevelIIQuotes';
import { OrderTicket } from '../../features/market/components/OrderTicket';
import { motion } from 'framer-motion';

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

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: { y: 0, opacity: 1 }
};

export const MarketPage: React.FC = () => {
  const [currentSymbol] = useState('AAPL');
  
  // Establish unified real-time connection
  const { isConnected } = useDataIntegration({ symbols: [currentSymbol] });

  return (
    <Box sx={{ p: 2, height: 'calc(100vh - 64px)', overflow: 'hidden', position: 'relative' }}>
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        style={{ height: '100%' }}
      >
        <Grid container spacing={2} sx={{ height: '100%' }}>
          {/* Left Column: Analytics & Chain */}
          <Grid item xs={12} lg={8.5} sx={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 2 }}>
            {/* Main Chart Card */}
            <motion.div variants={itemVariants} style={{ flex: 1, minHeight: 0 }}>
              <Box className="stitch-card" sx={{ height: '100%', p: 0, position: 'relative' }}>
                 <Box className="stitch-dots-container" sx={{ opacity: 0.05 }} />
                 <Box className="stitch-slanted-header" sx={{ bgcolor: stitchTokens.colors.abstract.indigo }}>LIVE_PRICE_TRAJECTORY // {currentSymbol}</Box>
                 <Box sx={{ p: 1, height: 'calc(100% - 32px)' }}>
                    <Suspense fallback={<LoadingFallback />}>
                      <LivePriceChart symbol={currentSymbol} />
                    </Suspense>
                 </Box>
              </Box>
            </motion.div>

            {/* Options Chain Card */}
            <motion.div variants={itemVariants} style={{ flex: 1, minHeight: 0 }}>
              <Box className="stitch-card" sx={{ height: '100%', p: 0, position: 'relative' }}>
                 <Box className="stitch-slanted-header" sx={{ bgcolor: stitchTokens.colors.abstract.teal }}>DERIVATIVE_CHAIN_MATRIX // DEC_2024</Box>
                 <Box sx={{ height: 'calc(100% - 32px)', overflow: 'auto' }}>
                    <Suspense fallback={<LoadingFallback />}>
                      <OptionsChain symbol={currentSymbol} />
                    </Suspense>
                 </Box>
              </Box>
            </motion.div>
          </Grid>

          {/* Right Column: Execution & Depth */}
          <Grid item xs={12} lg={3.5} sx={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 2 }}>
            <motion.div variants={itemVariants} style={{ flex: 1.5, minHeight: 0 }}>
               <DOMLadder symbol={currentSymbol} />
            </motion.div>

            <motion.div variants={itemVariants} style={{ flex: 1, minHeight: 0 }}>
               <LevelIIQuotes symbol={currentSymbol} />
            </motion.div>

            <motion.div variants={itemVariants} style={{ flex: 1, minHeight: 0 }}>
               <OrderTicket symbol={currentSymbol} />
            </motion.div>
          </Grid>
        </Grid>
      </motion.div>

      {/* Connection Toast (Overlay) */}
      <AnimatePresence>
        {!isConnected && (
          <motion.div
            initial={{ y: 50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 50, opacity: 0 }}
            style={{ position: 'fixed', bottom: 24, right: 24, zIndex: 1000 }}
          >
            <Box className="stitch-card" sx={{ 
              p: 2, 
              bgcolor: 'rgba(255, 46, 126, 0.1)',
              border: '1px solid #ff2e7e',
              borderLeft: '4px solid #ff2e7e'
            }}>
              <Typography className="stitch-label" sx={{ color: '#ff2e7e', fontWeight: 900 }}>
                PIPELINE_ERROR // RECONNECTING_SUBSYSTEM...
              </Typography>
            </Box>
          </motion.div>
        )}
      </AnimatePresence>
    </Box>
  );
};

export default MarketPage;
