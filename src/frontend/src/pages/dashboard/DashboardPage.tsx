import React, { lazy, Suspense } from 'react';
import {
  Box,
  Typography,
  Grid,
  Stack,
  alpha,
  CircularProgress,
} from '@mui/material';
import { usePricingStore } from '../../store/usePricingStore';
import type { PricingState } from '../../store/usePricingStore';
import { stitchTokens } from '../../theme/stitch-tokens';
import { DeepInferenceEngine } from '../../features/dashboard/components/DeepInferenceEngine';
import { RiskExposureGrid } from '../../features/dashboard/components/RiskExposureGrid';
import { motion } from 'framer-motion';

// Lazy loaded components
const LivePriceChart = lazy(() =>
  import('../../features/charts/components/LivePriceChart').then(m => ({ default: m.LivePriceChart }))
);

const LoadingFallback = () => (
  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', p: 4 }}>
    <CircularProgress size={20} aria-label="Loading section" sx={{ color: stitchTokens.colors.primary }} />
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

const KpiCard: React.FC<{ label: string; value: string; gradient: string; prefix?: string; index: number }> = ({ label, value, gradient, prefix, index }) => (
  <motion.div variants={itemVariants}>
    <Box className="stitch-card" sx={{ p: 2, position: 'relative', overflow: 'hidden' }}>
       <Box sx={{ 
         position: 'absolute', top: 0, right: 0, width: '40%', height: '100%', 
         background: gradient, opacity: 0.15, 
         clipPath: 'polygon(100% 0, 0 0, 100% 100%)',
         zIndex: 0
       }} />
       <Box sx={{ position: 'relative', zIndex: 1 }}>
          <Typography className="stitch-label" sx={{ fontSize: '9px', mb: 1, color: '#fff', opacity: 0.6 }}>{label}</Typography>
          <Stack direction="row" alignItems="baseline" spacing={0.5}>
             <Typography className="stitch-mono" sx={{ fontSize: '22px', fontWeight: 950, letterSpacing: '-1px' }}>
               {prefix}{value}
             </Typography>
          </Stack>
       </Box>
       {/* Decorative Shard */}
       <Box 
         className="stitch-abstract-shard" 
         sx={{ bottom: -10, left: '20%', width: 40, height: 4, background: gradient, opacity: 0.8 }} 
       />
    </Box>
  </motion.div>
);

export const DashboardPage: React.FC = () => {
  const portfolioTotal = usePricingStore((state: PricingState) => state.portfolioTotal);
  
  return (
    <Box sx={{ p: 2, height: 'calc(100vh - 64px)', overflow: 'auto', position: 'relative' }}>
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <Grid container spacing={2}>
          {/* Top KPI Row */}
          {[
            { label: 'NET_DELTA_EXPOSURE', value: '+0.428', gradient: stitchTokens.colors.abstract.teal },
            { label: 'THETA_DECAY_RATE', value: '-1.24k', gradient: stitchTokens.colors.abstract.purple },
            { label: 'VEGA_SENSITIVITY', value: '4.52k', gradient: stitchTokens.colors.abstract.indigo },
            { label: 'TOTAL_LIQUIDITY', value: '1.2M', gradient: stitchTokens.colors.abstract.orange, prefix: '$' },
          ].map((kpi, i) => (
            <Grid item xs={12} sm={6} md={3} key={kpi.label}>
               <KpiCard {...kpi} index={i} />
            </Grid>
          ))}

          {/* Intelligence Row */}
          <Grid item xs={12} lg={4}>
             <motion.div variants={itemVariants}>
                <DeepInferenceEngine />
             </motion.div>
          </Grid>
          <Grid item xs={12} lg={4}>
             <motion.div variants={itemVariants}>
                <RiskExposureGrid />
             </motion.div>
          </Grid>
          <Grid item xs={12} lg={4}>
             <motion.div variants={itemVariants}>
                <Box className="stitch-card" sx={{ height: '100%', p: 0, position: 'relative' }}>
                   {/* Abstract Geometric Decoration */}
                   <Box className="stitch-abstract-shard" sx={{ top: 20, right: 20, width: 60, height: 60, bgcolor: 'rgba(255,255,255,0.03)', clipPath: stitchTokens.geometry.shard }} />
                   
                   <Box className="stitch-slanted-header" sx={{ bgcolor: stitchTokens.colors.abstract.purple, border: 'none' }}>STRATEGY_ALLOCATION_MATRIX</Box>
                   <Box sx={{ p: 2.5, display: 'flex', flexDirection: 'column', gap: 2 }}>
                      {[
                        { name: 'NEUTRAL_CONDOR_v4', weight: 45, color: '#00FFA3' },
                        { name: 'BLACK_SWAN_HEDGE', weight: 25, color: '#A855F7' },
                        { name: 'INCOME_OVERLAY', weight: 30, color: '#3B82F6' },
                      ].map(strat => (
                        <Box key={strat.name}>
                           <Stack direction="row" justifyContent="space-between" sx={{ mb: 0.8 }}>
                              <Typography sx={{ fontSize: '10px', fontWeight: 900, letterSpacing: '1px' }}>{strat.name}</Typography>
                              <Typography className="stitch-mono" sx={{ fontSize: '10px', fontWeight: 900, color: strat.color }}>{strat.weight}%</Typography>
                           </Stack>
                           <Box sx={{ height: 2, width: '100%', bgcolor: 'rgba(255,255,255,0.05)', position: 'relative' }}>
                              <motion.div 
                                initial={{ width: 0 }}
                                animate={{ width: `${strat.weight}%` }}
                                transition={{ duration: 1, delay: 0.5 }}
                                style={{ height: '100%', backgroundColor: strat.color, boxShadow: `0 0 10px ${strat.color}` }} 
                              />
                           </Box>
                        </Box>
                      ))}
                   </Box>
                </Box>
             </motion.div>
          </Grid>

          {/* Observation Deck */}
          <Grid item xs={12}>
             <motion.div variants={itemVariants}>
                <Box className="stitch-card" sx={{ height: 500, p: 0, position: 'relative' }}>
                   <Box className="stitch-dots-container" sx={{ opacity: 0.05 }} />
                   <Box className="stitch-slanted-header" sx={{ bgcolor: '#1a1a1a' }}>TEMPORAL_TRAJECTORY // GLOBAL_INDICES</Box>
                   <Box sx={{ p: 1, height: 'calc(100% - 32px)' }}>
                      <Suspense fallback={<LoadingFallback />}>
                         <LivePriceChart symbol="SPX" />
                      </Suspense>
                   </Box>
                </Box>
             </motion.div>
          </Grid>

          {/* Recent Alerts Table */}
          <Grid item xs={12}>
             <motion.div variants={itemVariants}>
                <Box className="stitch-card" sx={{ p: 0 }}>
                   <Box className="stitch-banner-orange" style={{ fontSize: '10px', width: 'fit-content', transform: 'scale(1.1) translateX(20px)' }}>SYSTEM_LOGS // REAL-TIME_TELEMETRY</Box>
                   <Box sx={{ p: 1 }}>
                      {[
                        { time: '14:22:01', type: 'SIGNAL', msg: 'ML Model detected bearish divergence on AAPL 15m' },
                        { time: '14:20:45', type: 'EXEC', msg: 'Filled 500 contracts SPY 450P @ 1.45' },
                        { time: '14:18:12', type: 'RISK', msg: 'Vega threshold exceeded on NVDA portfolio' },
                      ].map((log, i) => (
                        <Box key={i} sx={{ 
                           display: 'flex', 
                           alignItems: 'center',
                           p: '10px 16px', 
                           borderBottom: '1px solid rgba(255,255,255,0.03)',
                           '&:hover': { bgcolor: 'rgba(255,255,255,0.03)' }
                        }}>
                           <Typography className="stitch-mono" sx={{ fontSize: '10px', color: '#a9abb1', width: 80 }}>{log.time}</Typography>
                           <Box sx={{ 
                             px: 1, py: 0.2, mr: 2, borderRadius: 0, 
                             bgcolor: log.type === 'RISK' ? 'rgba(255,46,126,0.1)' : 'rgba(0,255,163,0.1)',
                             border: `1px solid ${log.type === 'RISK' ? 'rgba(255,46,126,0.2)' : 'rgba(0,255,163,0.2)'}`
                           }}>
                              <Typography sx={{ 
                                 fontSize: '8px', fontWeight: 900,
                                 color: log.type === 'RISK' ? '#ff2e7e' : stitchTokens.colors.primary 
                              }}>
                                 {log.type}
                              </Typography>
                           </Box>
                           <Typography sx={{ fontSize: '11px', fontWeight: 600, color: 'rgba(255,255,255,0.8)' }}>{log.msg}</Typography>
                        </Box>
                      ))}
                   </Box>
                </Box>
             </motion.div>
          </Grid>
        </Grid>
      </motion.div>
    </Box>
  );
};

export default DashboardPage;
