import React from 'react';
import { Box, Typography, Stack, alpha } from '@mui/material';
import { stitchTokens } from '../../../theme/stitch-tokens';
import { motion } from 'framer-motion';

export const PortfolioHealth: React.FC = () => {
  return (
    <Box className="stitch-card" sx={{ p: 3, position: 'relative', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
       {/* Dot Matrix Layer */}
       <Box className="stitch-dots-container" sx={{ opacity: 0.05 }} />
       
       <Box sx={{ position: 'relative', zIndex: 1 }}>
          <Typography className="stitch-label" sx={{ mb: 1, fontSize: '10px', color: stitchTokens.colors.primary, letterSpacing: '2px', fontWeight: 900 }}>SYSTEMIC_PORTFOLIO_INTEGRITY_INDEX</Typography>
          <Stack direction="row" alignItems="baseline" spacing={1}>
             <Typography variant="h3" sx={{ fontWeight: 950, color: '#fff', fontSize: '3rem', letterSpacing: '-2px' }}>
               92.42
             </Typography>
             <Typography className="stitch-mono" sx={{ fontSize: '14px', color: 'rgba(255,255,255,0.4)', fontWeight: 700 }}>/ 100.00</Typography>
          </Stack>
          <Box className="stitch-banner-orange" style={{ mt: 1, fontSize: '8px', padding: '2px 12px' }}>OPERATIONAL_EFFICIENCY_OPTIMIZED</Box>
       </Box>
       
       {/* Sophisticated Radial Gauge */}
       <Box sx={{ position: 'relative', width: 120, height: 120, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <svg width="120" height="120" viewBox="0 0 120 120">
             <circle 
                cx="60" cy="60" r="54" 
                fill="none" 
                stroke="rgba(255,255,255,0.05)" 
                strokeWidth="2" 
             />
             <motion.circle 
                cx="60" cy="60" r="54" 
                fill="none" 
                stroke={stitchTokens.colors.primary} 
                strokeWidth="4" 
                strokeDasharray="339.29"
                initial={{ strokeDashoffset: 339.29 }}
                animate={{ strokeDashoffset: 339.29 * (1 - 0.9242) }}
                transition={{ duration: 1.5, ease: "easeOut" }}
                strokeLinecap="square"
                style={{ filter: `drop-shadow(0 0 8px ${stitchTokens.colors.primary})` }}
             />
             {/* Decorative ticks */}
             {[...Array(8)].map((_, i) => (
                <line 
                   key={i}
                   x1="60" y1="12" x2="60" y2="18" 
                   stroke="rgba(255,255,255,0.2)" 
                   strokeWidth="1" 
                   transform={`rotate(${i * 45} 60 60)`}
                />
             ))}
          </svg>
          <Box sx={{ position: 'absolute', textAlign: 'center' }}>
             <Typography className="stitch-mono" sx={{ fontSize: '12px', fontWeight: 950, color: stitchTokens.colors.primary }}>v4.2.0</Typography>
             <Typography sx={{ fontSize: '7px', opacity: 0.4, fontWeight: 900 }}>STABLE</Typography>
          </Box>
          
          {/* Abstract Geometric Decoration */}
          <Box className="stitch-abstract-shard" sx={{ position: 'absolute', top: -10, right: -10, width: 30, height: 30, bgcolor: 'rgba(0,255,163,0.05)', clipPath: stitchTokens.geometry.shard }} />
       </Box>
    </Box>
  );
};
