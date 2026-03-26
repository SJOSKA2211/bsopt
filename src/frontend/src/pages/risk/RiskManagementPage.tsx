import React from 'react';
import { Box, Grid, Stack, Typography, alpha } from '@mui/material';
import { stitchTokens } from '../../theme/stitch-tokens';
import { PortfolioHealth } from '../../features/risk/components/PortfolioHealth';
import { BlackSwanStressTest } from '../../features/risk/components/BlackSwanStressTest';
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

const RiskManagementPage: React.FC = () => {
  return (
    <Box sx={{ p: 2, height: 'calc(100vh - 64px)', overflow: 'auto', position: 'relative' }}>
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <Grid container spacing={2}>
          {/* Header Section */}
          <Grid size={{ xs: 12 }}>
             <motion.div variants={itemVariants}>
                <PortfolioHealth />
             </motion.div>
          </Grid>

          {/* Risk Analysis Grid */}
          <Grid size={{ xs: 12, lg: 8 }}>
             <motion.div variants={itemVariants}>
                <Box className="stitch-card" sx={{ height: 500, p: 0, position: 'relative' }}>
                   <Box className="stitch-dots-container" sx={{ opacity: 0.05 }} />
                   <Box className="stitch-slanted-header" sx={{ bgcolor: stitchTokens.colors.abstract.indigo }}>QUANT_RISK_MANIFOLD // P&L_SURFACE_v3.2</Box>
                   <Box sx={{ p: 2, height: 'calc(100% - 32px)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <Box sx={{ 
                        width: '100%', 
                        height: '100%', 
                        border: '1px solid rgba(255,255,255,0.05)', 
                        position: 'relative', 
                        bgcolor: 'rgba(0,0,0,0.2)',
                        overflow: 'hidden'
                      }}>
                         {/* Abstract Geometric Decoration */}
                         <Box className="stitch-abstract-shard float-animation" sx={{ top: '20%', left: '30%', width: 200, height: 200, bgcolor: 'rgba(0, 255, 163, 0.05)', clipPath: stitchTokens.geometry.shard }} />
                         <Typography className="stitch-label" sx={{ position: 'relative', zIndex: 1, textAlign: 'center', mt: '20%', opacity: 0.4, fontSize: '10px', fontWeight: 900 }}>
                            [ HD_RISK_HEATMAP_INITIALIZING... ]<br/>
                            GPU_ACCELERATED_PARALLEL_SCAN: ACTIVE
                         </Typography>
                      </Box>
                   </Box>
                </Box>
             </motion.div>
          </Grid>

          <Grid size={{ xs: 12, lg: 4 }}>
             <motion.div variants={itemVariants}>
                <BlackSwanStressTest />
             </motion.div>
          </Grid>

          {/* Detailed Metrics */}
          <Grid size={{ xs: 12, md: 6 }}>
             <motion.div variants={itemVariants}>
                <Box className="stitch-card" sx={{ p: 0 }}>
                   <Box className="stitch-banner-orange" style={{ transform: 'scale(1.1) translateX(24px)', marginBottom: -10, position: 'relative', zIndex: 1, width: 'fit-content' }}>HEDGE_SUB_ORCHESTRATOR</Box>
                   <Box className="stitch-slanted-header" sx={{ mt: 1, bgcolor: '#1a1a1a' }}>ACTIVE_DERIVATIVE_PROTECTION</Box>
                   <Box sx={{ p: 0 }}>
                      {[
                        { symbol: 'SPY_241220_P_450', delta: '-124.5', type: 'PUT', status: 'ACTIVE' },
                        { symbol: 'VIX_250117_C_25', delta: '+45.2', type: 'CALL', status: 'WAITING' },
                        { symbol: 'TLT_250321_L_95', delta: '+12.0', type: 'EQUITY', status: 'ACTIVE' },
                      ].map((hedge, i) => (
                        <Box key={i} sx={{ display: 'flex', alignItems: 'center', p: '12px 16px', borderBottom: '1px solid rgba(255,255,255,0.03)', '&:hover': { bgcolor: 'rgba(255,255,255,0.02)' } }}>
                           <Box sx={{ flex: 1 }}>
                              <Typography sx={{ fontSize: '10px', fontWeight: 950, color: '#fff', letterSpacing: '0.5px' }}>{hedge.symbol}</Typography>
                              <Typography className="stitch-label" sx={{ fontSize: '7px', opacity: 0.5 }}>{hedge.type} // STATUS: {hedge.status}</Typography>
                           </Box>
                           <Box sx={{ flex: 1, textAlign: 'right' }}>
                              <Typography className="stitch-mono" sx={{ fontSize: '11px', fontWeight: 900, color: hedge.delta.includes('-') ? '#ff2e7e' : stitchTokens.colors.primary }}>
                                 {hedge.delta} Δ
                              </Typography>
                              <Box sx={{ display: 'inline-block', width: 40, height: 2, bgcolor: 'rgba(255,255,255,0.1)', mt: 0.5 }}>
                                 <Box sx={{ height: '100%', width: '70%', bgcolor: hedge.delta.includes('-') ? '#ff2e7e' : stitchTokens.colors.primary }} />
                              </Box>
                           </Box>
                        </Box>
                      ))}
                   </Box>
                </Box>
             </motion.div>
          </Grid>

          <Grid size={{ xs: 12, md: 6 }}>
             <motion.div variants={itemVariants}>
                <Box className="stitch-card" sx={{ p: 0, position: 'relative' }}>
                   <Box className="stitch-slanted-header">MARGIN_LATENCY_MONITOR // CROSS_COLLATERAL</Box>
                   <Box sx={{ p: 2.5 }}>
                      <Stack spacing={3}>
                         <Box>
                            <Stack direction="row" justifyContent="space-between" sx={{ mb: 1.2 }}>
                               <Typography className="stitch-label" sx={{ fontSize: '9px', fontWeight: 900 }}>MARGIN_UTILIZATION_RATIO</Typography>
                               <Typography className="stitch-mono" sx={{ fontSize: '11px', fontWeight: 900, color: stitchTokens.colors.primary }}>34.22%</Typography>
                            </Stack>
                            <Box sx={{ height: 2, width: '100%', bgcolor: 'rgba(255,255,255,0.05)', position: 'relative' }}>
                               <motion.div 
                                 initial={{ width: 0 }}
                                 animate={{ width: '34.22%' }}
                                 transition={{ duration: 1, delay: 0.8 }}
                                 style={{ height: '100%', backgroundColor: stitchTokens.colors.primary, boxShadow: `0 0 10px ${stitchTokens.colors.primary}` }} 
                               />
                            </Box>
                         </Box>
                         
                         <Box>
                            <Stack direction="row" justifyContent="space-between" sx={{ mb: 1.2 }}>
                               <Typography className="stitch-label" sx={{ fontSize: '9px', fontWeight: 900 }}>LIQUIDITY_RESERVE_FACTOR</Typography>
                               <Typography className="stitch-mono" sx={{ fontSize: '11px', fontWeight: 900, color: stitchTokens.colors.secondary }}>8.45x</Typography>
                            </Stack>
                            <Box sx={{ height: 2, width: '100%', bgcolor: 'rgba(255,255,255,0.05)', position: 'relative' }}>
                               <motion.div 
                                 initial={{ width: 0 }}
                                 animate={{ width: '85%' }}
                                 transition={{ duration: 1, delay: 1 }}
                                 style={{ height: '100%', backgroundColor: stitchTokens.colors.secondary, boxShadow: `0 0 10px ${stitchTokens.colors.secondary}` }} 
                               />
                            </Box>
                         </Box>
    
                         <Box sx={{ mt: 1, p: 2, bgcolor: 'rgba(168, 85, 247, 0.03)', border: `1px solid ${alpha(stitchTokens.colors.secondary, 0.1)}`, position: 'relative' }}>
                            <Box className="stitch-abstract-shard" sx={{ top: 0, left: 0, width: 4, height: '100%', bgcolor: stitchTokens.colors.secondary }} />
                            <Typography sx={{ fontSize: '10px', fontWeight: 700, lineHeight: 1.5, color: 'rgba(255,255,255,0.7)', letterSpacing: '0.2px' }}>
                               <Box component="span" sx={{ color: stitchTokens.colors.secondary, fontWeight: 900 }}>THREAT_LEVEL: NOMINAL.</Box> Concentration in high-beta assets has shifted +4.2% since previous epoch. Re-calibration recommended if volatility exceeds 2.5σ.
                            </Typography>
                         </Box>
                      </Stack>
                   </Box>
                </Box>
             </motion.div>
          </Grid>
        </Grid>
      </motion.div>
    </Box>
  );
};

export default RiskManagementPage;
