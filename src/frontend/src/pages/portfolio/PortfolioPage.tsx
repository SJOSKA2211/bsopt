import React from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Stack,
  Chip,
  Box,
  alpha,
  useTheme,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Button,
} from '@mui/material';
import { motion } from 'framer-motion';
import {
  PieChart as AllocationIcon,
} from '@mui/icons-material';
import { PortfolioSummary } from '../../features/portfolio/components/PortfolioSummary';
import { usePortfolio } from '../../features/portfolio/hooks/usePortfolio';

// KPI constants removed - now dynamic

// Simple SVG donut chart
const DonutChart: React.FC = () => {
  const theme = useTheme();
  const segments = [
    { label: 'AAPL', pct: 28, color: theme.palette.financial.qfd.quantum },
    { label: 'SPY', pct: 22, color: theme.palette.financial.qfd.nebula },
    { label: 'QQQ', pct: 18, color: theme.palette.financial.qfd.electrum },
    { label: 'NVDA', pct: 15, color: theme.palette.info.main },
    { label: 'Cash', pct: 17, color: theme.palette.text.disabled },
  ];

  let cumulative = 0;
  const r = 70, cx = 90, cy = 90;
  const paths = segments.map((seg) => {
    const start = cumulative;
    cumulative += seg.pct;
    const startAngle = (start / 100) * 2 * Math.PI - Math.PI / 2;
    const endAngle = (cumulative / 100) * 2 * Math.PI - Math.PI / 2;
    const x1 = cx + r * Math.cos(startAngle);
    const y1 = cy + r * Math.sin(startAngle);
    const x2 = cx + r * Math.cos(endAngle);
    const y2 = cy + r * Math.sin(endAngle);
    const large = seg.pct > 50 ? 1 : 0;
    return { ...seg, d: `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2} Z` };
  });

  return (
    <Box>
      <svg role="img" aria-label="Portfolio Allocation Donut Chart" viewBox="0 0 180 180" style={{ width: '100%', maxWidth: 180 }}>
        {paths.map((p) => (
          <path
            key={p.label}
            d={p.d}
            fill={p.color}
            opacity={0.85}
            style={{ transition: 'opacity 0.2s', cursor: 'pointer' }}
            onMouseEnter={(e) => ((e.target as SVGPathElement).style.opacity = '1')}
            onMouseLeave={(e) => ((e.target as SVGPathElement).style.opacity = '0.85')}
          />
        ))}
        {/* Hole */}
        <circle cx={cx} cy={cy} r={42} fill="#0f172a" />
        <text x={cx} y={cy - 4} textAnchor="middle" fill="#f8fafc" fontSize="12" fontWeight="bold">AUM</text>
        <text x={cx} y={cy + 14} textAnchor="middle" fill="#10b981" fontSize="11">$48.4k</text>
      </svg>
      <Stack spacing={0.75} sx={{ mt: 1 }}>
        {segments.map(s => (
          <Stack key={s.label} direction="row" justifyContent="space-between" alignItems="center">
            <Stack direction="row" spacing={1} alignItems="center">
              <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: s.color, flexShrink: 0 }} />
              <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 600 }}>{s.label}</Typography>
            </Stack>
            <Typography variant="caption" sx={{ color: 'text.disabled', fontFamily: '"JetBrains Mono", monospace' }}>
              {s.pct}%
            </Typography>
          </Stack>
        ))}
      </Stack>
    </Box>
  );
};



export const PortfolioPage: React.FC = () => {
  const { data: portfolioData } = usePortfolio();

  const positions = portfolioData?.positions || [];
  
  const kpiCards = [
    { 
      label: 'Total Portfolio', 
      value: `$${(portfolioData?.totalValue || 0).toLocaleString()}`, 
      sub: `${portfolioData?.dailyPnLPercent || 0 >= 0 ? '+' : ''}${portfolioData?.dailyPnLPercent || 0}% today`, 
      type: 'quantum', 
      positive: (portfolioData?.dailyPnLPercent || 0) >= 0 
    },
    { 
      label: 'Total P&L', 
      value: `$${(portfolioData?.dailyPnL || 0).toLocaleString()}`, 
      sub: 'Real-time update', 
      type: 'quantum', 
      positive: (portfolioData?.dailyPnL || 0) >= 0 
    },
    { 
      label: 'Available Balance', 
      value: `$${(portfolioData?.balance || 0).toLocaleString()}`, 
      sub: 'Cash on hand', 
      type: 'nebula', 
      positive: true 
    },
    { 
      label: 'Frozen Capital', 
      value: `$${(portfolioData?.frozen_capital || 0).toLocaleString()}`, 
      sub: 'Margin requirements', 
      type: 'electrum', 
      positive: null 
    },
  ];

  return (
    <Container maxWidth="xl" sx={{ mt: 2, pb: 6 }}>
      {/* Header omitted for brevity in chunk - same as before */}
      {/* KPI cards */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
      >
        <Grid container spacing={3} sx={{ mb: 4 }}>
          {kpiCards.map((kpi) => {
            const accentColor = (theme.palette.financial.qfd as Record<string, string>)[kpi.type] || theme.palette.primary.main;
            return (
              <Grid key={kpi.label} size={{ xs: 12, sm: 6, lg: 3 }}>
                <motion.div whileHover={{ translateY: -5 }} transition={{ duration: 0.2 }}>
                  <Paper
                    className="stat-card"
                    sx={{
                      p: 3,
                      borderRadius: 6,
                      background: `linear-gradient(135deg, ${alpha('#0f172a', 0.6)}, ${alpha('#0f172a', 0.2)})`,
                      backdropFilter: 'blur(40px) saturate(200%)',
                      border: `1px solid ${alpha(accentColor, 0.15)}`,
                      position: 'relative',
                      overflow: 'hidden',
                      height: '100%'
                    }}
                  >
                    <Box sx={{ position: 'absolute', top: 0, left: 0, width: '100%', height: 3, background: `linear-gradient(90deg, transparent, ${accentColor}, transparent)`, opacity: 0.5 }} />
                    <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 800, letterSpacing: '0.12em', textTransform: 'uppercase' }}>
                      {kpi.label}
                    </Typography>
                    <Typography
                      variant="h2"
                      sx={{
                        fontWeight: 900,
                        my: 1,
                        fontFamily: 'JetBrains Mono',
                        color: kpi.positive ? accentColor : 'text.primary',
                        fontSize: '1.6rem',
                        letterSpacing: '-0.02em'
                      }}
                    >
                      {kpi.value}
                    </Typography>
                    <Typography variant="caption" sx={{ color: kpi.positive ? accentColor : kpi.positive === false ? 'error.main' : 'warning.main', fontWeight: 800 }}>
                      {kpi.sub}
                    </Typography>
                  </Paper>
                </motion.div>
              </Grid>
            );
          })}
        </Grid>
      </motion.div>

      {/* Charts row ... */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid size={{ xs: 12, lg: 4 }} className="slide-up" style={{ animationDelay: '0.1s' }}>
          <Paper
            sx={{
              p: 3,
              height: 380,
              borderRadius: 6,
              background: `linear-gradient(135deg, ${alpha('#0f172a', 0.6)}, ${alpha('#0f172a', 0.2)})`,
              backdropFilter: 'blur(40px) saturate(200%)',
              border: `1px solid ${alpha('#f8fafc', 0.08)}`,
            }}
          >
            <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 3 }}>
              <AllocationIcon sx={{ color: 'secondary.main', fontSize: 20 }} aria-label="Allocation Chart" />
              <Typography variant="h3" sx={{ fontWeight: 900, fontFamily: 'Outfit', fontSize: '1.25rem' }}>
                Allocation
              </Typography>
            </Stack>
            <DonutChart />
          </Paper>
        </Grid>
        {/* ... Other charts ... */}
      </Grid>

      {/* Positions row */}
      <Grid container spacing={3}>
        <Grid size={{ xs: 12, lg: 4 }}>
           <PortfolioSummary />
        </Grid>
        <Grid size={{ xs: 12, lg: 8 }} className="slide-up">
          <Paper
            sx={{
              overflow: 'hidden',
              borderRadius: 6,
              background: `linear-gradient(135deg, ${alpha('#0f172a', 0.6)}, ${alpha('#0f172a', 0.2)})`,
              backdropFilter: 'blur(40px) saturate(200%)',
              border: `1px solid ${alpha('#f8fafc', 0.08)}`,
            }}
          >
            <Stack direction="row" spacing={2} alignItems="center" sx={{ p: 3, pb: 2 }}>
              <Typography variant="h5" sx={{ fontWeight: 900, fontFamily: 'Outfit' }}>Open Positions</Typography>
              <Chip
                label={`${positions.length} active units`}
                size="small"
                sx={{
                  height: 24,
                  fontSize: '0.7rem',
                  fontWeight: 800,
                  bgcolor: alpha('#7B68EE', 0.1),
                  color: '#7B68EE',
                  border: `1px solid ${alpha('#7B68EE', 0.2)}`,
                  borderRadius: 1.5
                }}
              />
            </Stack>
            <Table size="small">
              <TableHead>
                <TableRow sx={{ bgcolor: alpha('#fff', 0.02) }}>
                  {['Symbol', 'Qty', 'Entry', 'Current', 'P&L', 'P&L%', 'Action'].map((h) => (
                    <TableCell key={h} sx={{ color: 'text.secondary', fontWeight: 900, fontSize: '0.7rem', textTransform: 'uppercase' }}>{h}</TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {positions.map((p: any, idx: number) => (
                  <TableRow key={p.id || idx}>
                    <TableCell sx={{ fontWeight: 900, color: 'primary.main' }}>{p.symbol || p.contract_symbol}</TableCell>
                    <TableCell sx={{ fontFamily: 'JetBrains Mono' }}>{p.quantity}</TableCell>
                    <TableCell sx={{ fontFamily: 'JetBrains Mono' }}>${(p.entry_price || 0).toFixed(2)}</TableCell>
                    <TableCell sx={{ fontFamily: 'JetBrains Mono' }}>${(p.entry_price || 0).toFixed(2)}</TableCell>
                    <TableCell sx={{ fontWeight: 900, color: 'success.main' }}>$0.00</TableCell>
                    <TableCell sx={{ fontWeight: 900, color: 'success.main' }}>0.0%</TableCell>
                    <TableCell>
                      <Button size="small" variant="contained" sx={{ bgcolor: alpha('#f43f5e', 0.1), color: '#f43f5e' }}>LIQUIDATE</Button>
                    </TableCell>
                  </TableRow>
                ))}
                {!positions.length && (
                  <TableRow>
                     <TableCell colSpan={7} sx={{ textAlign: 'center', py: 4, color: 'text.disabled' }}>No active positions found.</TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default PortfolioPage;
