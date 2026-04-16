import { Box, Typography, Stack, Button, alpha } from '@mui/material';
import { motion } from 'framer-motion';

export const BlackSwanStressTest: React.FC = () => {
  const scenarios = [
    { name: '2008_GFC_RECURRENCE', impact: '-18.42%', risk: 'CRITICAL', color: '#ff4d4d' },
    { name: '1987_BLACK_MONDAY_v2', impact: '-22.15%', risk: 'FATAL', color: '#ff2e7e' },
    { name: 'COVID_FLASH_CRASH_SYMMETRY', impact: '-12.50%', risk: 'HIGH', color: '#ff8a00' },
    { name: 'INT_RATE_DISLOC_SCAN', impact: '-5.24%', risk: 'MODERATE', color: '#ffcc00' }
  ];

  return (
    <Box className="stitch-card" sx={{ height: '100%', p: 0, position: 'relative', overflow: 'hidden', border: '1px solid rgba(255, 46, 126, 0.2)' }}>
      <Box className="stitch-dots-container" sx={{ opacity: 0.1, backgroundImage: `radial-gradient(${alpha('#ff2e7e', 0.2)} 1px, transparent 0)` }} />
      <Box className="stitch-slanted-header" sx={{ bgcolor: '#ff2e7e', color: 'white', fontWeight: 950, letterSpacing: '1px' }}>
         TAIL_RISK_SIMULATION // BLACK_SWAN
      </Box>
      <Box sx={{ p: 2, position: 'relative', zIndex: 1 }}>
        <Stack spacing={1}>
          {scenarios.map((s, i) => (
            <motion.div
               key={i}
               initial={{ x: -20, opacity: 0 }}
               animate={{ x: 0, opacity: 1 }}
               transition={{ delay: i * 0.1 + 0.5 }}
            >
               <Box sx={{ 
                 p: '10px 14px', 
                 bgcolor: 'rgba(0,0,0,0.4)', 
                 border: `1px solid ${alpha(s.color, 0.15)}`,
                 position: 'relative',
                 overflow: 'hidden',
                 '&:hover': { bgcolor: alpha(s.color, 0.05), border: `1px solid ${alpha(s.color, 0.4)}` }
               }}>
                  <Box sx={{ position: 'absolute', top: 0, left: 0, width: 2, height: '100%', bgcolor: s.color }} />
                  <Stack direction="row" justifyContent="space-between" alignItems="center">
                     <Box>
                       <Typography sx={{ fontSize: '10px', fontWeight: 900, color: '#fff', letterSpacing: '0.5px' }}>{s.name}</Typography>
                       <Typography className="stitch-label" sx={{ fontSize: '7px', fontWeight: 800, color: s.color, opacity: 0.8 }}>
                         THREAT_LEVEL: {s.risk}
                       </Typography>
                     </Box>
                     <Typography className="stitch-mono" sx={{ fontSize: '13px', fontWeight: 950, color: s.color, textShadow: `0 0 10px ${alpha(s.color, 0.5)}` }}>
                       {s.impact}
                     </Typography>
                  </Stack>
               </Box>
            </motion.div>
          ))}
        </Stack>
        
        <Box sx={{ mt: 2, p: 1, bgcolor: 'rgba(255, 46, 126, 0.05)', border: '1px dashed rgba(255, 46, 126, 0.3)', mb: 2 }}>
           <Typography sx={{ fontSize: '8px', fontWeight: 700, color: '#ff2e7e', textAlign: 'center' }}>
              WARNING: PORTFOLIO_VAR EXCEEDS 2.5% IN ALL TAIL SCENARIOS
           </Typography>
        </Box>

        <Button 
          fullWidth 
          variant="outlined"
          sx={{ 
            borderRadius: 0, 
            height: 36,
            borderColor: '#ff2e7e', 
            color: '#ff2e7e', 
            fontWeight: 950,
            fontSize: '9px',
            letterSpacing: '1px',
            '&:hover': { bgcolor: alpha('#ff2e7e', 0.1), borderColor: '#ff2e7e' }
          }}
        >
          EXECUTE_STRESS_MANIFOLD_SCAN
        </Button>
      </Box>
    </Box>
  );
};
