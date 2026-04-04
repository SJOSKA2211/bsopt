import React, { lazy, Suspense } from 'react';
import {
  Box,
  Typography,
  Stack,
  CircularProgress,
} from '@mui/material';
import { usePricingStore } from '../../store/usePricingStore';
import type { PricingState } from '../../store/usePricingStore';
import { DeepInferenceEngine } from '../../features/dashboard/components/DeepInferenceEngine';
import { RiskExposureGrid } from '../../features/dashboard/components/RiskExposureGrid';
import { motion } from 'framer-motion';
import { AnimatedCard } from '../../components/common/AnimatedCard';

// Lazy loaded components
const LivePriceChart = lazy(() =>
  import('../../features/charts/components/LivePriceChart').then(m => ({ default: m.LivePriceChart }))
);

const LoadingFallback = () => (
  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', p: 4 }}>
    <CircularProgress size={20} sx={{ color: 'var(--accent-mint)' }} />
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

const KpiCard: React.FC<{ label: string; value: string; color: string; prefix?: string; index: number }> = ({ label, value, color, prefix, index }) => (
  <AnimatedCard delay={index * 0.05} sx={{ p: 3 }}>
     <Stack spacing={1}>
        <Typography className="label-secondary" sx={{ fontSize: '11px', opacity: 0.6 }}>{label}</Typography>
        <Stack direction="row" alignItems="baseline" spacing={0.5}>
           <Typography className="data-mono" sx={{ fontSize: '28px', fontWeight: 800, color: 'var(--text-primary)' }}>
             {prefix}{value}
           </Typography>
        </Stack>
        <Box sx={{ height: 2, width: 40, bgcolor: color, borderRadius: 2 }} />
     </Stack>
  </AnimatedCard>
);

export const DashboardPage: React.FC = () => {
  return (
    <Box sx={{ p: '24px', height: 'calc(100vh - 64px)', overflow: 'auto' }}>
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="bento-grid"
      >
        {/* KPI Row */}
        {[
          { label: 'NET_DELTA', value: '+0.428', color: 'var(--accent-mint)' },
          { label: 'THETA_DECAY', value: '-1.24k', color: 'var(--accent-purple)' },
          { label: 'VEGA_SENS', value: '4.52k', color: 'var(--accent-teal)' },
          { label: 'LIQUIDITY', value: '1.2M', color: 'var(--accent-amber)', prefix: '$' },
        ].map((kpi, i) => (
          <Box key={kpi.label} sx={{ gridColumn: { xs: 'span 12', sm: 'span 6', lg: 'span 3' } }}>
             <KpiCard {...kpi} index={i} />
          </Box>
        ))}

        {/* Intelligence Layer */}
        <Box sx={{ gridColumn: { xs: 'span 12', lg: 'span 4' } }}>
           <AnimatedCard delay={0.2} sx={{ height: '100%' }}>
              <Typography className="label-secondary" sx={{ mb: 3 }}>DEEP_INFERENCE_ENGINE</Typography>
              <DeepInferenceEngine />
           </AnimatedCard>
        </Box>
        
        <Box sx={{ gridColumn: { xs: 'span 12', lg: 'span 4' } }}>
           <AnimatedCard delay={0.25} sx={{ height: '100%' }}>
              <Typography className="label-secondary" sx={{ mb: 3 }}>RISK_EXPOSURE_GRID</Typography>
              <RiskExposureGrid />
           </AnimatedCard>
        </Box>

        <Box sx={{ gridColumn: { xs: 'span 12', lg: 'span 4' } }}>
           <AnimatedCard delay={0.3} sx={{ height: '100%' }}>
              <Typography className="label-secondary" sx={{ mb: 3 }}>STRATEGY_ALLOCATION</Typography>
              <Stack spacing={3}>
                 {[
                   { name: 'NEUTRAL_CONDOR_v4', weight: 45, color: 'var(--accent-mint)' },
                   { name: 'BLACK_SWAN_HEDGE', weight: 25, color: 'var(--accent-purple)' },
                   { name: 'INCOME_OVERLAY', weight: 30, color: 'var(--accent-teal)' },
                 ].map(strat => (
                   <Box key={strat.name}>
                      <Stack direction="row" justifyContent="space-between" sx={{ mb: 1 }}>
                         <Typography sx={{ fontSize: '11px', fontWeight: 600 }}>{strat.name}</Typography>
                         <Typography className="data-mono" sx={{ fontSize: '11px', color: strat.color }}>{strat.weight}%</Typography>
                      </Stack>
                      <Box sx={{ height: 4, width: '100%', bgcolor: 'rgba(255,255,255,0.05)', borderRadius: 2, overflow: 'hidden' }}>
                         <motion.div 
                           initial={{ width: 0 }}
                           animate={{ width: `${strat.weight}%` }}
                           transition={{ duration: 1, delay: 0.5 }}
                           style={{ height: '100%', backgroundColor: strat.color }} 
                         />
                      </Box>
                   </Box>
                 ))}
              </Stack>
           </AnimatedCard>
        </Box>

        {/* Observation Deck */}
        <Box sx={{ gridColumn: 'span 12' }}>
           <AnimatedCard delay={0.4} sx={{ height: 540, p: 0 }}>
              <Box sx={{ p: 3, borderBottom: '1px solid var(--bento-card-border)' }}>
                 <Typography className="label-secondary">TEMPORAL_TRAJECTORY // GLOBAL_INDICES</Typography>
              </Box>
              <Box sx={{ p: 2, height: 'calc(100% - 65px)' }}>
                 <Suspense fallback={<LoadingFallback />}>
                    <LivePriceChart symbol="SPX" />
                 </Suspense>
              </Box>
           </AnimatedCard>
        </Box>

        {/* Signals Telemetry */}
        <Box sx={{ gridColumn: 'span 12' }}>
           <AnimatedCard delay={0.5} sx={{ p: 0 }}>
              <Box sx={{ p: 3, borderBottom: '1px solid var(--bento-card-border)', display: 'flex', justifyContent: 'space-between' }}>
                 <Typography className="label-secondary">SIGNAL_TELEMETRY</Typography>
                 <Box className="status-pill healthy">LIVE_FEED</Box>
              </Box>
              <Box sx={{ p: 1 }}>
                 {[
                   { time: '14:22:01', type: 'SIGNAL', msg: 'ML Model detected bearish divergence on AAPL 15m' },
                   { time: '14:20:45', type: 'EXEC', msg: 'Filled 500 contracts SPY 450P @ 1.45' },
                   { time: '14:18:12', type: 'RISK', msg: 'Vega threshold exceeded on NVDA portfolio' },
                 ].map((log, i) => (
                   <Box key={i} sx={{ 
                      display: 'flex', 
                      alignItems: 'center',
                      p: '16px 24px', 
                      borderBottom: i === 2 ? 'none' : '1px solid rgba(255,255,255,0.03)',
                      '&:hover': { bgcolor: 'rgba(255,255,255,0.02)' }
                   }}>
                      <Typography className="data-mono" sx={{ fontSize: '11px', color: 'var(--text-secondary)', width: 100 }}>{log.time}</Typography>
                      <Box className={`status-pill ${log.type === 'RISK' ? 'critical' : 'healthy'}`} sx={{ mr: 3 }}>
                         {log.type}
                      </Box>
                      <Typography sx={{ fontSize: '13px', fontWeight: 500 }}>{log.msg}</Typography>
                   </Box>
                 ))}
              </Box>
           </AnimatedCard>
        </Box>
      </motion.div>
    </Box>
  );
};

export default DashboardPage;

export default DashboardPage;
