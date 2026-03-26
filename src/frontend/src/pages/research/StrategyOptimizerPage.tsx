import React from 'react';
import { Box, Grid, Stack, Typography, Button, alpha } from '@mui/material';
import { stitchTokens } from '../../theme/stitch-tokens';
import { OptimizationControls } from '../../features/research/components/OptimizationControls';
import { OptimalConfigCard } from '../../features/research/components/OptimalConfigCard';
import { SweepResultsTable } from '../../features/research/components/SweepResultsTable';
import VolatilitySurface3D from '../../features/options/components/VolatilitySurface3D';
import { motion } from 'framer-motion';

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

const StrategyOptimizerPage: React.FC = () => {
  return (
    <Box sx={{ p: 2, height: 'calc(100vh - 64px)', overflow: 'auto', position: 'relative' }}>
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <Grid container spacing={2}>
          {/* Header Shard */}
          <Grid size={{ xs: 12 }}>
             <motion.div variants={itemVariants}>
                <Box className="stitch-card" sx={{ p: '24px 32px', position: 'relative', overflow: 'hidden' }}>
                   <Box className="stitch-dots-container" sx={{ opacity: 0.1 }} />
                   <Box className="stitch-abstract-shard float-animation" sx={{ top: -40, right: -40, width: 250, height: 250, bgcolor: alpha(stitchTokens.colors.primary, 0.05), clipPath: stitchTokens.geometry.shard }} />
                   
                   <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ position: 'relative', zIndex: 1 }}>
                      <Box>
                         <Typography className="stitch-label" sx={{ color: stitchTokens.colors.primary, mb: 1, letterSpacing: '2px', fontWeight: 900 }}>
                            STRATEGY_GENESIS // GEOMETRIC_OPTIMIZER_v4.2
                         </Typography>
                         <Stack direction="row" alignItems="baseline" spacing={2}>
                            <Typography className="stitch-mono" sx={{ fontSize: '32px', fontWeight: 950, letterSpacing: '-1px' }}>
                               AAPL // BULL_CALL_SPREAD
                            </Typography>
                            <Box className="stitch-banner-orange" style={{ transform: 'skewX(-15deg)', padding: '2px 12px', fontSize: '10px' }}>RUN_SCAN_SEQUENCE_042</Box>
                         </Stack>
                      </Box>
                      <Button 
                         variant="contained"
                         sx={{ 
                            height: 48,
                            bgcolor: stitchTokens.colors.primary, 
                            color: '#000', 
                            px: 6, 
                            borderRadius: 0,
                            fontSize: '11px',
                            fontWeight: 950,
                            letterSpacing: '1.5px',
                            boxShadow: `0 0 25px ${alpha(stitchTokens.colors.primary, 0.4)}`,
                            '&:hover': { bgcolor: alpha(stitchTokens.colors.primary, 0.9) }
                         }}
                      >
                         EXECUTE_HEURISTIC_OPTIMIZATION
                      </Button>
                   </Stack>
                </Box>
             </motion.div>
          </Grid>

          {/* Controls Bar */}
          <Grid size={{ xs: 12 }}>
             <motion.div variants={itemVariants}>
                <Box className="stitch-card" sx={{ p: 0, position: 'relative', overflow: 'hidden' }}>
                   <Box className="stitch-slanted-header" sx={{ bgcolor: 'rgba(255,255,255,0.02)', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                      INPUT_PARAMETERS // CONSTRAINTS
                   </Box>
                   <Box sx={{ p: 2 }}>
                      <OptimizationControls />
                   </Box>
                </Box>
             </motion.div>
          </Grid>

          {/* 3D Manifold */}
          <Grid size={{ xs: 12, lg: 8 }}>
             <motion.div variants={itemVariants}>
                <Box className="stitch-card" sx={{ height: 500, p: 0, position: 'relative', overflow: 'hidden' }}>
                   <Box className="stitch-dots-container" sx={{ opacity: 0.03 }} />
                   <Box className="stitch-slanted-header" sx={{ bgcolor: stitchTokens.colors.abstract.purple }}>OPTIMIZATION_MANIFOLD // 3D_HYPER_SURFACE_SCAN</Box>
                   <Box sx={{ p: '8px', height: 'calc(100% - 32px)', position: 'relative' }}>
                      <VolatilitySurface3D symbol="AAPL" />
                      <Box sx={{ position: 'absolute', bottom: 12, left: 12, p: '4px 12px', bgcolor: 'rgba(0,0,0,0.6)', border: '1px solid rgba(255,255,255,0.1)' }}>
                         <Typography className="stitch-mono" sx={{ fontSize: '8px', color: stitchTokens.colors.secondary, fontWeight: 900 }}>COORD_SYSTEM: CARTESIAN_v3 // SCALE: 1.2x</Typography>
                      </Box>
                   </Box>
                </Box>
             </motion.div>
          </Grid>

          {/* Optimal Configs */}
          <Grid size={{ xs: 12, lg: 4 }}>
             <motion.div variants={itemVariants}>
                <Box className="stitch-card" sx={{ height: 500, p: 0, position: 'relative', overflow: 'hidden' }}>
                   <Box className="stitch-dots-container" sx={{ opacity: 0.05 }} />
                   <Box className="stitch-slanted-header" sx={{ bgcolor: '#1a1a1a' }}>TOP_RANKED_CONFIGURATIONS</Box>
                   <Box sx={{ p: 2, height: 'calc(100% - 32px)', overflow: 'auto', position: 'relative', zIndex: 1 }}>
                      <Stack spacing={1.5}>
                         {[
                           { id: 1, strike1: 190.0, strike2: 195.0, change: '+24.52%', score: '2.42' },
                           { id: 2, strike1: 187.5, strike2: 192.5, change: '+21.24%', score: '2.18' },
                           { id: 3, strike1: 192.5, strike2: 197.5, change: '+19.95%', score: '1.95' },
                           { id: 4, strike1: 190.0, strike2: 200.0, change: '+28.12%', score: '1.82' },
                         ].map(config => (
                            <OptimalConfigCard key={config.id} {...config} />
                         ))}
                      </Stack>
                   </Box>
                </Box>
             </motion.div>
          </Grid>

          {/* Bottom Sweep Results */}
          <Grid size={{ xs: 12 }}>
             <motion.div variants={itemVariants}>
                <Box className="stitch-card" sx={{ p: 0, overflow: 'hidden' }}>
                   <Box className="stitch-slanted-header" sx={{ bgcolor: stitchTokens.colors.abstract.indigo }}>PARAMETER_SWEEP_SEQUENCE // DETAILED_TELEMETRY</Box>
                   <Box sx={{ p: 0 }}>
                      <SweepResultsTable />
                   </Box>
                </Box>
             </motion.div>
          </Grid>
        </Grid>
      </motion.div>
    </Box>
  );
};

export default StrategyOptimizerPage;
