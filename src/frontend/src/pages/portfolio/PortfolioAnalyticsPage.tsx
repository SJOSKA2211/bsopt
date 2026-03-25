import React from 'react';
import { Box, Grid, Stack, Typography, alpha } from '@mui/material';
import { stitchTokens } from '../../theme/stitch-tokens';
import { EquityCurveChart } from '../../features/portfolio/components/EquityCurveChart';
import { ActivePositionsTable } from '../../features/portfolio/components/ActivePositionsTable';
import { RecentTradeActivity } from '../../features/portfolio/components/RecentTradeActivity';

const PortfolioAnalyticsPage: React.FC = () => {
  return (
    <Box sx={{ p: 2, height: 'calc(100vh - 72px)', overflow: 'auto' }}>
      <Grid container spacing={2}>
        {/* Key Metrics Row */}
        <Grid item xs={12}>
           <Box className="stitch-card" sx={{ p: 2 }}>
              <Grid container spacing={4}>
                 {[
                   { label: 'NET LIQUIDATION', value: '$254,120.42', color: stitchTokens.colors.primary },
                   { label: 'DAILY P&L', value: '+$3,420.12', color: stitchTokens.colors.primary },
                   { label: 'MAINTENANCE MARGIN', value: '$42,500.00', color: stitchTokens.colors.secondary },
                   { label: 'EXCESS LIQUIDITY', value: '$211,620.42', color: stitchTokens.colors.tertiary },
                 ].map(m => (
                   <Grid item xs={12} sm={6} md={3} key={m.label}>
                      <Typography className="stitch-label" sx={{ fontSize: '9px', mb: 0.5 }}>{m.label}</Typography>
                      <Typography className="stitch-mono" sx={{ fontSize: '18px', fontWeight: 900, color: m.color }}>{m.value}</Typography>
                   </Grid>
                 ))}
              </Grid>
           </Box>
        </Grid>

        {/* Performance Chart */}
        <Grid item xs={12} lg={8}>
           <Box className="stitch-card" sx={{ height: 450, p: 0 }}>
              <Box className="stitch-slanted-header">EQUITY CURVE // PERFORMANCE TELEMETRY</Box>
              <Box sx={{ p: 1, height: 'calc(100% - 32px)' }}>
                 <EquityCurveChart />
              </Box>
           </Box>
        </Grid>

        {/* Portfolio Greeks / Summary */}
        <Grid item xs={12} lg={4}>
           <Box className="stitch-card" sx={{ height: 450, p: 0 }}>
              <Box className="stitch-slanted-header" sx={{ bgcolor: stitchTokens.colors.secondary }}>PORTFOLIO GREEKS // AGGREGATED</Box>
              <Box sx={{ p: 2 }}>
                 <Stack spacing={3}>
                    {[
                      { label: 'TOTAL DELTA', value: '+42.5 Δ', percent: 65, color: stitchTokens.colors.primary },
                      { label: 'TOTAL GAMMA', value: '+1.24 Γ', percent: 45, color: stitchTokens.colors.secondary },
                      { label: 'TOTAL THETA', value: '-245.1 Θ', percent: 80, color: '#ff4d4d' },
                      { label: 'TOTAL VEGA', value: '+142.0 V', percent: 30, color: stitchTokens.colors.tertiary },
                    ].map(g => (
                      <Box key={g.label}>
                         <Stack direction="row" justifyContent="space-between" sx={{ mb: 0.5 }}>
                            <Typography className="stitch-label" sx={{ fontSize: '9px' }}>{g.label}</Typography>
                            <Typography className="stitch-mono" sx={{ fontSize: '12px', fontWeight: 900 }}>{g.value}</Typography>
                         </Stack>
                         <Box sx={{ height: 4, width: '100%', bgcolor: 'rgba(255,255,255,0.05)' }}>
                            <Box sx={{ height: '100%', width: `${g.percent}%`, bgcolor: g.color }} />
                         </Box>
                      </Box>
                    ))}
                 </Stack>
              </Box>
           </Box>
        </Grid>

        {/* Positions Section */}
        <Grid item xs={12}>
           <Box className="stitch-card" sx={{ p: 0 }}>
              <Box className="stitch-slanted-header">ACTIVE POSITIONS // INVENTORY MONITOR</Box>
              <Box sx={{ p: 0 }}>
                 <ActivePositionsTable />
              </Box>
           </Box>
        </Grid>

        {/* Bottom Activity Section */}
        <Grid item xs={12}>
           <Box className="stitch-card" sx={{ p: 0 }}>
              <Box className="stitch-slanted-header" sx={{ bgcolor: stitchTokens.colors.tertiary }}>RECENT TRADE ACTIVITY // EXECUTION LOG</Box>
              <Box sx={{ p: 0 }}>
                 <RecentTradeActivity />
              </Box>
           </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default PortfolioAnalyticsPage;
