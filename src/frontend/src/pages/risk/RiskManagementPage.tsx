import React from 'react';
import { Box, Grid, Stack, Typography, alpha } from '@mui/material';
import { stitchTokens } from '../../theme/stitch-tokens';
import { PortfolioHealth } from '../../features/risk/components/PortfolioHealth';
import { BlackSwanStressTest } from '../../features/risk/components/BlackSwanStressTest';

const RiskManagementPage: React.FC = () => {
  return (
    <Box sx={{ p: 2, height: 'calc(100vh - 72px)', overflow: 'auto' }}>
      <Grid container spacing={2}>
        {/* Header Section */}
        <Grid item xs={12}>
           <PortfolioHealth />
        </Grid>

        {/* Risk Analysis Grid */}
        <Grid item xs={12} lg={8}>
           <Box className="stitch-card" sx={{ height: 500, p: 0 }}>
              <Box className="stitch-slanted-header">2D STRESS MANIFOLD // P&L SURFACE</Box>
              <Box sx={{ p: 2, height: 'calc(100% - 32px)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                 <Box sx={{ width: '100%', height: '100%', border: '1px solid rgba(255,255,255,0.05)', position: 'relative', bgcolor: 'rgba(0,0,0,0.2)' }}>
                    {/* Placeholder for the complex canvas heatmap */}
                    <Typography className="stitch-label" sx={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', opacity: 0.3 }}>
                       [ CANVAS MANIFOLD RENDERING ]
                    </Typography>
                 </Box>
              </Box>
           </Box>
        </Grid>

        <Grid item xs={12} lg={4}>
           <BlackSwanStressTest />
        </Grid>

        {/* Detailed Metrics */}
        <Grid item xs={12} md={6}>
           <Box className="stitch-card" sx={{ p: 0 }}>
              <Box className="stitch-slanted-header" sx={{ bgcolor: stitchTokens.colors.primary, color: 'black' }}>ACTIVE HEDGES // PROTECTIVE REBALANCING</Box>
              <Box sx={{ p: 0 }}>
                 {[
                   { symbol: 'SPY Puts', delta: '-124.5', expiry: 'DEC 24', status: 'ACTIVE' },
                   { symbol: 'VIX Calls', delta: '+45.2', expiry: 'JAN 25', status: 'WAITING' },
                   { symbol: 'TLT Long', delta: '+12.0', expiry: 'MAR 25', status: 'ACTIVE' },
                 ].map((hedge, i) => (
                   <Box key={i} sx={{ display: 'flex', p: 1.5, borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                      <Box sx={{ flex: 1 }}>
                         <Typography sx={{ fontSize: '11px', fontWeight: 800 }}>{hedge.symbol}</Typography>
                         <Typography className="stitch-label" sx={{ fontSize: '8px' }}>{hedge.expiry}</Typography>
                      </Box>
                      <Box sx={{ flex: 1, textAlign: 'right' }}>
                         <Typography className="stitch-mono" sx={{ fontSize: '11px', fontWeight: 900, color: hedge.delta.includes('-') ? '#ff4d4d' : stitchTokens.colors.primary }}>
                            {hedge.delta} Δ
                         </Typography>
                         <Typography sx={{ fontSize: '8px', fontWeight: 900, color: hedge.status === 'ACTIVE' ? stitchTokens.colors.primary : '#a9abb1' }}>
                            {hedge.status}
                         </Typography>
                      </Box>
                   </Box>
                 ))}
              </Box>
           </Box>
        </Grid>

        <Grid item xs={12} md={6}>
           <Box className="stitch-card" sx={{ p: 0 }}>
              <Box className="stitch-slanted-header">MARGIN MONITOR // COLLATERAL HEALTH</Box>
              <Box sx={{ p: 2 }}>
                 <Stack spacing={2}>
                    <Box>
                       <Stack direction="row" justifyContent="space-between" sx={{ mb: 0.5 }}>
                          <Typography className="stitch-label" sx={{ fontSize: '9px' }}>MARGIN UTILIZATION</Typography>
                          <Typography className="stitch-mono" sx={{ fontSize: '11px', fontWeight: 800 }}>34.2%</Typography>
                       </Stack>
                       <Box sx={{ height: 6, width: '100%', bgcolor: 'rgba(255,255,255,0.05)', borderRadius: 3 }}>
                          <Box sx={{ height: '100%', width: '34.2%', bgcolor: stitchTokens.colors.primary, borderRadius: 3 }} />
                       </Box>
                    </Box>
                    
                    <Box>
                       <Stack direction="row" justifyContent="space-between" sx={{ mb: 0.5 }}>
                          <Typography className="stitch-label" sx={{ fontSize: '9px' }}>LIQUIDITY COVERAGE</Typography>
                          <Typography className="stitch-mono" sx={{ fontSize: '11px', fontWeight: 800 }}>8.4x</Typography>
                       </Stack>
                       <Box sx={{ height: 6, width: '100%', bgcolor: 'rgba(255,255,255,0.05)', borderRadius: 3 }}>
                          <Box sx={{ height: '100%', width: '85%', bgcolor: stitchTokens.colors.secondary, borderRadius: 3 }} />
                       </Box>
                    </Box>

                    <Box sx={{ mt: 1, p: 1.5, bgcolor: alpha(stitchTokens.colors.secondary, 0.05), border: `1px dashed ${alpha(stitchTokens.colors.secondary, 0.3)}` }}>
                       <Typography sx={{ fontSize: '10px', fontWeight: 500, lineHeight: 1.4 }}>
                          Warning: Liquidity coverage ratio (LCR) is healthy, but concentration in high-beta assets has increased by 4.2% in the last session.
                       </Typography>
                    </Box>
                 </Stack>
              </Box>
           </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default RiskManagementPage;
