import React from 'react';
import { Box, Grid, Stack, Typography, alpha } from '@mui/material';
import { stitchTokens } from '../../theme/stitch-tokens';
import { EquityCurveChart } from '../../features/portfolio/components/EquityCurveChart';
import { ActivePositionsTable } from '../../features/portfolio/components/ActivePositionsTable';
import { RecentTradeActivity } from '../../features/portfolio/components/RecentTradeActivity';
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

const PortfolioAnalyticsPage: React.FC = () => {
  return (
    <Box sx={{ p: 2, height: 'calc(100vh - 64px)', overflow: 'auto', position: 'relative' }}>
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <Grid container spacing={2}>
          {/* Key Metrics Row */}
          <Grid size={{ xs: 12 }}>
             <motion.div variants={itemVariants}>
                <Box className="stitch-card" sx={{ p: '24px 32px', position: 'relative', overflow: 'hidden' }}>
                   <Box className="stitch-dots-container" sx={{ opacity: 0.05 }} />
                   <Grid container spacing={4}>
                      {[
                        { label: 'NET_LIQUIDATION_VALUE', value: '$254,120.42', color: stitchTokens.colors.primary, trend: '+0.85%' },
                        { label: 'DAILY_UNREALIZED_P&L', value: '+$3,420.12', color: stitchTokens.colors.primary, trend: '+1.4%' },
                        { label: 'MAINTENANCE_MARGIN_CAP', value: '$42,500.00', color: stitchTokens.colors.secondary, trend: 'NOMINAL' },
                        { label: 'EXCESS_LIQUIDITY_BUFFER', value: '$211,620.42', color: stitchTokens.colors.tertiary, trend: 'SECURE' },
                      ].map((m, idx) => (
                        <Grid size={{ xs: 12, sm: 6, md: 3 }} key={m.label}>
                           <Typography className="stitch-label" sx={{ fontSize: '9px', fontWeight: 900, mb: 1, letterSpacing: '1px', opacity: 0.6 }}>{m.label}</Typography>
                           <Typography className="stitch-mono" sx={{ fontSize: '22px', fontWeight: 950, color: '#fff', letterSpacing: '-1px' }}>{m.value}</Typography>
                           <Stack direction="row" spacing={1} sx={{ mt: 0.5 }}>
                              <Box sx={{ p: '2px 8px', bgcolor: alpha(m.color, 0.1), borderLeft: `2px solid ${m.color}` }}>
                                 <Typography sx={{ fontSize: '8px', fontWeight: 900, color: m.color }}>{m.trend}</Typography>
                              </Box>
                           </Stack>
                           {idx < 3 && (
                             <Box sx={{ display: { xs: 'none', md: 'block' }, position: 'absolute', right: 0, top: '20%', height: '60%', width: '1px', bgcolor: 'rgba(255,255,255,0.05)' }} />
                           )}
                        </Grid>
                      ))}
                   </Grid>
                </Box>
             </motion.div>
          </Grid>

          {/* Performance Chart */}
          <Grid size={{ xs: 12, lg: 8 }}>
             <motion.div variants={itemVariants}>
                <Box className="stitch-card" sx={{ height: 480, p: 0, position: 'relative' }}>
                   <Box className="stitch-slanted-header" sx={{ bgcolor: stitchTokens.colors.abstract.purple }}>EQUITY_CURVE_TELEMETRY // HISTORICAL_DATA_v8</Box>
                   <Box sx={{ p: 1, height: 'calc(100% - 32px)', position: 'relative' }}>
                      <Box sx={{ position: 'absolute', top: 20, right: 20, zIndex: 10 }}>
                         <Stack direction="row" spacing={1}>
                            {['1D', '1W', '1M', '3M', 'YTD', 'ALL'].map(t => (
                               <Box key={t} sx={{ p: '2px 8px', bgcolor: t === '3M' ? alpha(stitchTokens.colors.primary, 0.1) : 'rgba(0,0,0,0.3)', border: `1px solid ${t === '3M' ? stitchTokens.colors.primary : 'rgba(255,255,255,0.05)'}`, cursor: 'pointer' }}>
                                  <Typography sx={{ fontSize: '8px', fontWeight: 900, color: t === '3M' ? stitchTokens.colors.primary : 'rgba(255,255,255,0.4)' }}>{t}</Typography>
                               </Box>
                            ))}
                         </Stack>
                      </Box>
                      <EquityCurveChart />
                   </Box>
                </Box>
             </motion.div>
          </Grid>

          {/* Portfolio Greeks */}
          <Grid size={{ xs: 12, lg: 4 }}>
             <motion.div variants={itemVariants}>
                <Box className="stitch-card" sx={{ height: 480, p: 0, position: 'relative', overflow: 'hidden' }}>
                   <Box className="stitch-dots-container" sx={{ opacity: 0.03 }} />
                   <Box className="stitch-slanted-header">AGGREGATED_PORTFOLIO_GREEKS</Box>
                   <Box sx={{ p: 2.5 }}>
                      <Stack spacing={4}>
                        {[
                          { label: 'NET_DELTA_EXPOSURE', value: '+42.52 Δ', percent: 65, color: stitchTokens.colors.primary },
                          { label: 'GAMMA_ACCELERATION', value: '+1.240 Γ', percent: 45, color: stitchTokens.colors.secondary },
                          { label: 'THETA_DECAY_ABS', value: '-245.12 Θ', percent: 72, color: '#ff2e7e' },
                          { label: 'VEGA_VOL_SENSITIVITY', value: '+142.05 V', percent: 30, color: stitchTokens.colors.tertiary },
                        ].map(g => (
                          <Box key={g.label}>
                             <Stack direction="row" justifyContent="space-between" sx={{ mb: 1.2 }}>
                                <Typography className="stitch-label" sx={{ fontSize: '9px', fontWeight: 900 }}>{g.label}</Typography>
                                <Typography className="stitch-mono" sx={{ fontSize: '12px', fontWeight: 900, color: g.color }}>{g.value}</Typography>
                             </Stack>
                             <Box sx={{ height: 2, width: '100%', bgcolor: 'rgba(255,255,255,0.05)', position: 'relative' }}>
                                <motion.div 
                                  initial={{ width: 0 }}
                                  animate={{ width: `${g.percent}%` }}
                                  transition={{ duration: 1, delay: 0.5 }}
                                  style={{ height: '100%', backgroundColor: g.color, boxShadow: `0 0 10px ${alpha(g.color, 0.4)}` }} 
                                />
                             </Box>
                          </Box>
                        ))}
                      </Stack>
                      
                      <Box sx={{ mt: 5, p: 2, bgcolor: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)', position: 'relative' }}>
                         <Typography className="stitch-label" sx={{ fontSize: '8px', mb: 1 }}>GREEK_SYMMETRY_ANALysis</Typography>
                         <Typography sx={{ fontSize: '9px', fontWeight: 500, lineHeight: 1.6, color: 'rgba(255,255,255,0.6)' }}>
                            Portfolio is currently <Box component="span" sx={{ color: stitchTokens.colors.primary, fontWeight: 900 }}>DELTA_NEUTRAL_BIAS</Box>. Gamma exposure is concentrated in near-term expirations. Suggest re-balancing Vega exposure if IV exceeds 22%.
                         </Typography>
                         <Box className="stitch-abstract-shard" sx={{ bottom: -10, right: -10, width: 40, height: 40, bgcolor: alpha(stitchTokens.colors.primary, 0.05), clipPath: stitchTokens.geometry.shard }} />
                      </Box>
                   </Box>
                </Box>
             </motion.div>
          </Grid>

          {/* Positions Section */}
          <Grid size={{ xs: 12 }}>
             <motion.div variants={itemVariants}>
                <Box className="stitch-card" sx={{ p: 0, overflow: 'hidden' }}>
                   <Box className="stitch-banner-orange" style={{ transform: 'skewX(-20deg)', position: 'absolute', top: 4, right: 32, zIndex: 1, fontSize: '9px' }}>REAL_TIME_INVENTORY</Box>
                   <Box className="stitch-slanted-header" sx={{ bgcolor: '#121418', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>ACTIVE_POSITIONS_CORE</Box>
                   <Box sx={{ p: 0 }}>
                      <ActivePositionsTable />
                   </Box>
                </Box>
             </motion.div>
          </Grid>

          {/* Bottom Activity Section */}
          <Grid size={{ xs: 12 }}>
             <motion.div variants={itemVariants}>
                <Box className="stitch-card" sx={{ p: 0 }}>
                   <Box className="stitch-slanted-header" sx={{ bgcolor: stitchTokens.colors.abstract.indigo }}>RECENT_EXECUTION_SEQUENCE</Box>
                   <Box sx={{ p: 0 }}>
                      <RecentTradeActivity />
                   </Box>
                </Box>
             </motion.div>
          </Grid>
        </Grid>
      </motion.div>
    </Box>
  );
};

export default PortfolioAnalyticsPage;
