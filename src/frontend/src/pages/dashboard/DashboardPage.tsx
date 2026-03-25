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

// Lazy loaded components
const LivePriceChart = lazy(() =>
  import('../../features/charts/components/LivePriceChart').then(m => ({ default: m.LivePriceChart }))
);

const LoadingFallback = () => (
  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', p: 4 }}>
    <CircularProgress size={20} sx={{ color: stitchTokens.colors.primary }} />
  </Box>
);

const KpiCard: React.FC<{ label: string; value: string; color: string; prefix?: string }> = ({ label, value, color, prefix }) => (
  <Box className="stitch-card" sx={{ p: 2, borderLeft: `3px solid ${color}` }}>
     <Typography className="stitch-label" sx={{ fontSize: '9px', mb: 1 }}>{label}</Typography>
     <Stack direction="row" alignItems="baseline" spacing={0.5}>
        <Typography className="stitch-mono" sx={{ fontSize: '18px', fontWeight: 900 }}>
          {prefix}{value}
        </Typography>
     </Stack>
  </Box>
);

export const DashboardPage: React.FC = () => {
  const portfolioTotal = usePricingStore((state: PricingState) => state.portfolioTotal);
  
  return (
    <Box sx={{ p: 2, height: 'calc(100vh - 72px)', overflow: 'auto' }}>
      <Grid container spacing={2}>
        {/* Top KPI Row */}
        {[
          { label: 'NET DELTA', value: '+0.428', color: stitchTokens.colors.primary },
          { label: 'THETA DECAY', value: '-1.24k', color: stitchTokens.colors.secondary },
          { label: 'VEGA EXPOSURE', value: '4.52k', color: stitchTokens.colors.tertiary },
          { label: 'BUYING POWER', value: '1.2M', color: '#f5f6fc', prefix: '$' },
        ].map(kpi => (
          <Grid item xs={12} sm={6} md={3} key={kpi.label}>
             <KpiCard {...kpi} />
          </Grid>
        ))}

        {/* Intelligence Row */}
        <Grid item xs={12} lg={4}>
           <DeepInferenceEngine />
        </Grid>
        <Grid item xs={12} lg={4}>
           <RiskExposureGrid />
        </Grid>
        <Grid item xs={12} lg={4}>
           <Box className="stitch-card" sx={{ height: '100%', p: 0 }}>
              <Box className="stitch-slanted-header" sx={{ bgcolor: stitchTokens.colors.tertiary }}>STRATEGY ALLOCATION</Box>
              <Box sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                 {[
                   { name: 'Neutral Condor', weight: 45, color: stitchTokens.colors.primary },
                   { name: 'Tail Hedge', weight: 25, color: stitchTokens.colors.secondary },
                   { name: 'Income Overlay', weight: 30, color: stitchTokens.colors.tertiary },
                 ].map(strat => (
                   <Box key={strat.name}>
                      <Stack direction="row" justifyContent="space-between" sx={{ mb: 0.5 }}>
                         <Typography sx={{ fontSize: '10px', fontWeight: 700 }}>{strat.name}</Typography>
                         <Typography sx={{ fontSize: '10px', fontWeight: 900 }}>{strat.weight}%</Typography>
                      </Stack>
                      <Box sx={{ height: 4, width: '100%', bgcolor: 'rgba(255,255,255,0.05)' }}>
                         <Box sx={{ height: '100%', width: `${strat.weight}%`, bgcolor: strat.color }} />
                      </Box>
                   </Box>
                 ))}
              </Box>
           </Box>
        </Grid>

        {/* Observation Deck */}
        <Grid item xs={12}>
           <Box className="stitch-card" sx={{ height: 500, p: 0 }}>
              <Box className="stitch-slanted-header">TEMPORAL TRAJECTORY // GLOBAL INDEXES</Box>
              <Box sx={{ p: 1, height: 'calc(100% - 32px)' }}>
                 <Suspense fallback={<LoadingFallback />}>
                    <LivePriceChart symbol="SPX" />
                 </Suspense>
              </Box>
           </Box>
        </Grid>

        {/* Recent Alerts Table */}
        <Grid item xs={12}>
           <Box className="stitch-card" sx={{ p: 0 }}>
              <Box className="stitch-slanted-header">SYSTEM LOGS // REAL-TIME ALERTS</Box>
              <Box sx={{ p: 0 }}>
                 {[
                   { time: '14:22:01', type: 'SIGNAL', msg: 'ML Model detected bearish divergence on AAPL 15m' },
                   { time: '14:20:45', type: 'EXEC', msg: 'Filled 500 contracts SPY 450P @ 1.45' },
                   { time: '14:18:12', type: 'RISK', msg: 'Vega threshold exceeded on NVDA portfolio' },
                 ].map((log, i) => (
                   <Box key={i} sx={{ 
                      display: 'flex', 
                      p: '8px 16px', 
                      borderBottom: '1px solid rgba(255,255,255,0.03)',
                      '&:hover': { bgcolor: 'rgba(255,255,255,0.02)' }
                   }}>
                      <Typography className="stitch-mono" sx={{ fontSize: '10px', color: '#a9abb1', width: 80 }}>{log.time}</Typography>
                      <Typography className="stitch-label" sx={{ 
                         fontSize: '9px', 
                         width: 60, 
                         color: log.type === 'RISK' ? '#ff4d4d' : log.type === 'EXEC' ? stitchTokens.colors.primary : stitchTokens.colors.secondary 
                      }}>
                         [{log.type}]
                      </Typography>
                      <Typography sx={{ fontSize: '11px', fontWeight: 500 }}>{log.msg}</Typography>
                   </Box>
                 ))}
              </Box>
           </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DashboardPage;
