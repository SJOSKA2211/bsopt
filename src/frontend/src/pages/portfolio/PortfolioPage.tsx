import React, { useMemo } from 'react';
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
import type { Position } from '../../api/types';
import { PortfolioSummary } from '../../features/portfolio/components/PortfolioSummary';
import { usePortfolio } from '../../features/portfolio/hooks/usePortfolio';

// KPI constants removed - now dynamic

// Simple SVG donut chart
const DonutChart: React.FC<{ positions: Position[], totalValue: number }> = React.memo(({ positions, totalValue }) => {
  const theme = useTheme();
  
  const colors = useMemo(() => [
    theme.palette.info.main,
    theme.palette.secondary.main,
    theme.palette.warning.main,
    theme.palette.info.main,
    theme.palette.warning.main,
  ], [theme.palette.financial.qfd, theme.palette.info.main, theme.palette.warning.main]);

  const segments = useMemo(() => {
    const rawSegments = positions.length > 0 
      ? positions.slice(0, 4).map((p: Position, idx: number) => {
          const val = p.quantity * (p.current_price || p.entry_price || 0);
          const pct = totalValue > 0 ? Math.round((val / totalValue) * 100) : 0;
          return { label: p.symbol || p.contract_symbol, pct, color: colors[idx % colors.length] };
        })
      : [{ label: 'Cash', pct: 100, color: theme.palette.text.disabled }];
    
    const allocatedPct = rawSegments.reduce((sum: number, s: { pct: number }) => sum + s.pct, 0);
    if (allocatedPct < 100 && positions.length > 0) {
        rawSegments.push({ label: 'Cash', pct: 100 - allocatedPct, color: theme.palette.text.disabled });
    }
    return rawSegments;
  }, [positions, totalValue, colors, theme.palette.text.disabled]);
  
  const r = 70, cx = 90, cy = 90;

  const paths = useMemo(() => {
    let currentCumulative = 0;
    return segments.map((seg: { label?: string; pct: number; color: string }) => {
      const start = currentCumulative;
      currentCumulative += seg.pct;
      const startAngle = (start / 100) * 2 * Math.PI - Math.PI / 2;
      const endAngle =  (currentCumulative / 100) * 2 * Math.PI - Math.PI / 2;
      const x1 = cx + r * Math.cos(startAngle);
      const y1 = cy + r * Math.sin(startAngle);
      const x2 = cx + r * Math.cos(endAngle);
      const y2 = cy + r * Math.sin(endAngle);
      const large = seg.pct > 50 ? 1 : 0;
      return { ...seg, d: `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2} Z` };
    });
  }, [segments, cx, cy, r]);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <Box sx={{ position: 'relative', width: 220, height: 220 }}>
        <svg role="img" aria-label="Portfolio Allocation Donut Chart" viewBox="0 0 180 180" style={{ width: '100%', height: '100%' }}>
          {paths.map((p: { label?: string; pct: number; color: string; d: string }, idx: number) => (
            <motion.path
              key={p.label || idx}
              d={p.d}
              fill={p.color}
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 0.85 }}
              transition={{ duration: 0.8, delay: idx * 0.1, ease: "easeOut" }}
              whileHover={{ opacity: 1, scale: 1.02 }}
            />
          ))}
          <circle cx={cx} cy={cy} r={r * 0.6} fill={alpha('#0f172a', 0.8)} />
          <text x={cx} y={cy - 4} textAnchor="middle" fill="#f8fafc" fontSize="11" fontWeight="900" style={{ pointerEvents: 'none' }}>AUM</text>
          <text x={cx} y={cy + 14} textAnchor="middle" fill="#10b981" fontSize="13" fontWeight="900" style={{ pointerEvents: 'none', fontFamily: 'JetBrains Mono' }}>
            ${((totalValue || 0) / 1000).toFixed(1)}k
          </text>
        </svg>
      </Box>
      <Stack spacing={1} sx={{ mt: 2, width: '100%', px: 2 }}>
        {segments.map((s: { label?: string; pct: number; color: string }, idx: number) => (
          <motion.div
            key={s.label || idx}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5 + idx * 0.05 }}
          >
            <Stack direction="row" justifyContent="space-between" alignItems="center">
              <Stack direction="row" spacing={1.5} alignItems="center">
                <Box sx={{ width: 10, height: 10, borderRadius: 1, bgcolor: s.color, boxShadow: `0 0 10px ${alpha(s.color, 0.4)}` }} />
                <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 800, fontSize: '0.75rem' }}>{s.label}</Typography>
              </Stack>
              <Typography variant="caption" sx={{ color: 'text.primary', fontWeight: 900, fontFamily: 'JetBrains Mono' }}>
                {s.pct}%
              </Typography>
            </Stack>
          </motion.div>
        ))}
      </Stack>
    </Box>
  );
});

export const PortfolioPage: React.FC = () => {
  const theme = useTheme();
  const { data: portfolioData } = usePortfolio();
  const { totalValue = 0, dailyPnL = 0, dailyPnLPercent = 0, balance = 0, frozen_capital = 0, positions = [] } = portfolioData || {};
  
  const kpiCards = useMemo(() => [
    { 
      label: 'Total Portfolio', 
      value: `$${totalValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, 
      sub: `${dailyPnLPercent >= 0 ? '+' : ''}${dailyPnLPercent.toFixed(2)}% today`, 
      type: 'quantum', 
      positive: dailyPnLPercent >= 0 
    },
    { 
      label: 'Daily P&L', 
      value: `${dailyPnL >= 0 ? '+' : ''}$${Math.abs(dailyPnL).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, 
      sub: 'Production Real-time', 
      type: 'quantum', 
      positive: dailyPnL >= 0 
    },
    { 
      label: 'Available Balance', 
      value: `$${balance.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, 
      sub: 'Settled Liquidity', 
      type: 'nebula', 
      positive: true 
    },
    { 
      label: 'Frozen Capital', 
      value: `$${frozen_capital.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, 
      sub: 'Risk Requirement', 
      type: 'electrum', 
      positive: null as boolean | null
    },
  ], [totalValue, dailyPnL, dailyPnLPercent, balance, frozen_capital]);

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
          {kpiCards.map((kpi: { label: string; value: string; sub: string; type: string; positive: boolean | null }) => {
            const accentColor = (theme.palette.financial.qfd as Record<string, string>)[kpi.type] || theme.palette.primary.main;
            return (
              <Grid key={kpi.label} size={{xs: 12, sm: 6, lg: 3}}>
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
        <Grid size={{xs: 12, lg: 4}} className="slide-up" style={{ animationDelay: '0.1s' }}>
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
            <DonutChart positions={positions} totalValue={portfolioData?.totalValue || 0} />
          </Paper>
        </Grid>
        {/* ... Other charts ... */}
      </Grid>

      {/* Positions row */}
      <Grid container spacing={3}>
        <Grid size={{xs: 12, lg: 4}}>
           <PortfolioSummary />
        </Grid>
        <Grid size={{xs: 12, lg: 8}} className="slide-up">
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
                  {['Symbol', 'Qty', 'Entry', 'Current', 'P&L', 'P&L%', 'Action'].map((h: string) => (
                    <TableCell key={h} sx={{ color: 'text.secondary', fontWeight: 900, fontSize: '0.7rem', textTransform: 'uppercase' }}>{h}</TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {positions.map((p: Position, idx: number) => {
                  const currentPrice = p.current_price || p.entry_price || 0;
                  const pnl = (currentPrice - (p.entry_price || 0)) * p.quantity;
                  const pnlPct = p.entry_price > 0 ? ((currentPrice - p.entry_price) / p.entry_price) * 100 : 0;
                  const isPositive = pnl >= 0;

                  return (
                    <motion.tr
                      key={p.id || idx}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.2 + idx * 0.05 }}
                      style={{ borderBottom: `1px solid ${alpha(theme.palette.divider, 0.05)}` }}
                    >
                      <TableCell sx={{ fontWeight: 900, color: 'primary.main', borderBottom: 'none' }}>{p.symbol || p.contract_symbol}</TableCell>
                      <TableCell sx={{ fontFamily: 'JetBrains Mono', borderBottom: 'none' }}>{p.quantity}</TableCell>
                      <TableCell sx={{ fontFamily: 'JetBrains Mono', borderBottom: 'none' }}>${(p.entry_price || 0).toFixed(2)}</TableCell>
                      <TableCell sx={{ fontFamily: 'JetBrains Mono', borderBottom: 'none' }}>${currentPrice.toFixed(2)}</TableCell>
                      <TableCell sx={{ fontWeight: 900, color: isPositive ? 'success.main' : 'error.main', borderBottom: 'none' }}>
                        {isPositive ? '+' : ''}${pnl.toFixed(2)}
                      </TableCell>
                      <TableCell sx={{ fontWeight: 900, color: isPositive ? 'success.main' : 'error.main', borderBottom: 'none' }}>
                        <Box sx={{ 
                          px: 1, py: 0.5, borderRadius: 1.5, display: 'inline-block',
                          bgcolor: alpha(isPositive ? theme.palette.success.main : theme.palette.error.main, 0.1) 
                        }}>
                          {isPositive ? '+' : ''}{pnlPct.toFixed(2)}%
                        </Box>
                      </TableCell>
                      <TableCell sx={{ borderBottom: 'none' }}>
                        <Button 
                          size="small" 
                          variant="contained" 
                          sx={{ 
                            bgcolor: alpha('#f43f5e', 0.1), 
                            color: '#f43f5e',
                            fontWeight: 900,
                            '&:hover': { bgcolor: alpha('#f43f5e', 0.2) }
                          }}
                        >
                          LIQUIDATE
                        </Button>
                      </TableCell>
                    </motion.tr>
                  );
                })}
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
