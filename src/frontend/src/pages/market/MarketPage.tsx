import React, { lazy, Suspense, useState } from 'react';
import {
  Box,
  Typography,
  CircularProgress,
  Grid,
} from '@mui/material';
import { usePricingStore } from '../../store/usePricingStore';
import type { PricingState } from '../../store/usePricingStore';
import { useDataIntegration } from '../../hooks/useDataIntegration';
import { stitchTokens } from '../../theme/stitch-tokens';
import { DOMLadder } from '../../features/market/components/DOMLadder';
import { LevelIIQuotes } from '../../features/market/components/LevelIIQuotes';
import { OrderTicket } from '../../features/market/components/OrderTicket';
import { motion, AnimatePresence } from 'framer-motion';

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
  
  // Get live price from store
  const priceData = usePricingStore((state: PricingState) => state.prices[currentSymbol]);
  const livePrice = priceData?.price ?? 189.45;

  return (
    <div className="p-6 h-full overflow-hidden bg-bento-bg">
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="bento-grid h-full"
      >
        {/* Left Column: Analytics (spanning 8/12) */}
        <div className="col-span-12 lg:col-span-8 flex flex-col gap-6 min-h-0">
          <motion.div variants={itemVariants} className="flex-[1.2] min-h-0">
             <AnimatedCard className="h-full !p-0 border-white/5 overflow-hidden">
                <div className="p-4 border-b border-white/5 bg-white/2 flex justify-between items-center">
                   <h2 className="label-secondary opacity-40">LIVE_PRICE_TRAJECTORY // {currentSymbol}</h2>
                   <div className="status-pill healthy scale-75">REAL_TIME</div>
                </div>
                <div className="p-2 h-[calc(100%-56px)]">
                  <Suspense fallback={<LoadingFallback />}>
                    <LivePriceChart symbol={currentSymbol} />
                  </Suspense>
                </div>
             </AnimatedCard>
          </motion.div>

          <motion.div variants={itemVariants} className="flex-1 min-h-0">
             <AnimatedCard className="h-full !p-0 border-white/5 overflow-hidden">
                <div className="p-4 border-b border-white/5 bg-white/2">
                   <h2 className="label-secondary opacity-40 uppercase tracking-widest">DERIVATIVE_CHAIN_MATRIX // OPTION_SERIES</h2>
                </div>
                <div className="h-[calc(100%-56px)] overflow-auto">
                  <Suspense fallback={<LoadingFallback />}>
                    <OptionsChain symbol={currentSymbol} />
                  </Suspense>
                </div>
             </AnimatedCard>
          </motion.div>
        </div>

        {/* Right Column: Execution (spanning 4/12) */}
        <div className="col-span-12 lg:col-span-4 flex flex-col gap-6 min-h-0">
          <motion.div variants={itemVariants} className="flex-[1.5] min-h-0">
             <DOMLadder symbol={currentSymbol} currentPrice={livePrice} />
          </motion.div>

          <motion.div variants={itemVariants} className="flex-1 min-h-0">
             <LevelIIQuotes symbol={currentSymbol} />
          </motion.div>

          <motion.div variants={itemVariants} className="flex-1 min-h-0">
             <OrderTicket symbol={currentSymbol} />
          </motion.div>
        </div>
      </motion.div>

      {/* Connection Toast Overlay */}
      <AnimatePresence>
        {!isConnected && (
          <motion.div
            initial={{ y: 50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 50, opacity: 0 }}
            className="fixed bottom-6 right-6 z-[100]"
          >
            <div className="status-pill critical !py-3 !px-6 bg-red-500/10 border-red-500/40 text-red-400 font-black shadow-[0_0_20px_rgba(239,68,68,0.2)]">
               PIPELINE_ERROR // RECONNECTING_SUBSYSTEM...
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default MarketPage;
