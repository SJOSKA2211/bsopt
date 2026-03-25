import React from 'react';
import { Box, Grid, Stack, Typography, Button } from '@mui/material';
import { stitchTokens } from '../../theme/stitch-tokens';
import { OptimizationControls } from '../../features/research/components/OptimizationControls';
import { OptimalConfigCard } from '../../features/research/components/OptimalConfigCard';
import { SweepResultsTable } from '../../features/research/components/SweepResultsTable';
import VolatilitySurface3D from '../../features/options/components/VolatilitySurface3D';

const StrategyOptimizerPage: React.FC = () => {
  return (
    <Box sx={{ p: 2, height: 'calc(100vh - 72px)', overflow: 'auto' }}>
      <Grid container spacing={2}>
        {/* Header Shard */}
        <Grid item xs={12}>
           <Box className="stitch-card" sx={{ p: 2, position: 'relative', overflow: 'hidden' }}>
              <Box sx={{ 
                position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', 
                background: `linear-gradient(135deg, ${stitchTokens.colors.primary}11 0%, transparent 40%)`,
                pointerEvents: 'none'
              }} />
              <Stack direction="row" justifyContent="space-between" alignItems="center">
                 <Box>
                    <Typography className="stitch-label" sx={{ color: stitchTokens.colors.primary, mb: 0.5 }}>
                       STRATEGY // GEOMETRIC OPTIMIZER
                    </Typography>
                    <Typography className="stitch-mono" sx={{ fontSize: '24px', fontWeight: 900 }}>
                       AAPL // BULL CALL SPREAD <Box component="span" sx={{ opacity: 0.3 }}>// RUN_042</Box>
                    </Typography>
                 </Box>
                 <Button className="stitch-slanted-header" sx={{ 
                    bgcolor: stitchTokens.colors.primary, 
                    color: '#000', 
                    px: 4, 
                    py: 1, 
                    fontWeight: 900,
                    '&:hover': { bgcolor: stitchTokens.colors.primary, opacity: 0.9 }
                 }}>
                    EXECUTE OPTIMIZATION
                 </Button>
              </Stack>
           </Box>
        </Grid>

        {/* Controls Bar */}
        <Grid item xs={12}>
           <Box className="stitch-card" sx={{ p: 2, bgcolor: 'rgba(255,255,255,0.02)' }}>
              <OptimizationControls />
           </Box>
        </Grid>

        {/* 3D Manifold */}
        <Grid item xs={12} lg={8}>
           <Box className="stitch-card" sx={{ height: 500, p: 0 }}>
              <Box className="stitch-slanted-header">OPTIMIZATION MANIFOLD // 3D HYPER-SURFACE</Box>
              <Box sx={{ p: 1, height: 'calc(100% - 32px)' }}>
                 <VolatilitySurface3D symbol="AAPL" />
              </Box>
           </Box>
        </Grid>

        {/* Optimal Configs */}
        <Grid item xs={12} lg={4}>
           <Box className="stitch-card" sx={{ height: 500, p: 0 }}>
              <Box className="stitch-slanted-header" sx={{ bgcolor: stitchTokens.colors.secondary }}>OPTIMAL CONFIGS // TOP_RANKED</Box>
              <Box sx={{ p: 2, height: 'calc(100% - 32px)', overflow: 'auto' }}>
                 <Stack spacing={2}>
                    {[
                      { id: 1, strike1: 190, strike2: 195, change: '+24.5%', score: '2.42' },
                      { id: 2, strike1: 187.5, strike2: 192.5, change: '+21.2%', score: '2.18' },
                      { id: 3, strike1: 192.5, strike2: 197.5, change: '+19.9%', score: '1.95' },
                      { id: 4, strike1: 190, strike2: 200, change: '+28.1%', score: '1.82' },
                    ].map(config => (
                       <OptimalConfigCard key={config.id} {...config} />
                    ))}
                 </Stack>
              </Box>
           </Box>
        </Grid>

        {/* Bottom Sweep Results */}
        <Grid item xs={12}>
           <Box className="stitch-card" sx={{ p: 0 }}>
              <Box className="stitch-slanted-header" sx={{ bgcolor: stitchTokens.colors.tertiary }}>PARAMETER SWEEP // FULL RESULTS</Box>
              <Box sx={{ p: 0 }}>
                 <SweepResultsTable />
              </Box>
           </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default StrategyOptimizerPage;
