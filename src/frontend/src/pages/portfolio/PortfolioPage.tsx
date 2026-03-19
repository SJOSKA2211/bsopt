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
  TrendingUp as TrendingUpIcon,
} from '@mui/icons-material';
import { PortfolioSummary } from '../../features/portfolio/components/PortfolioSummary';
import { PnLExplainChart } from '../../features/risk/components/PnLExplainChart';

// ---- Mock open positions ----
const POSITIONS = [
  { symbol: 'AAPL', type: 'CALL', strike: '$185', expiry: '21 Mar', qty: 5, cost: '$4.20', price: '$6.88', pnl: +1340, pnlPct: +63.8, delta: 0.64 },
  { symbol: 'BTC', type: 'CRYPTO', strike: '-', expiry: '-', qty: 0.5, cost: '$62,000', price: '$68,500', pnl: +3250, pnlPct: +10.5, delta: 1.00 },
  { symbol: 'EUR/USD', type: 'FOREX', strike: '-', expiry: '-', qty: 100000, cost: '1.0850', price: '1.0920', pnl: +700, pnlPct: +0.65, delta: 1.00 },
  { symbol: 'SPY', type: 'CALL', strike: '$470', expiry: '18 Apr', qty: 10, cost: '$5.60', price: '$7.12', pnl: +1520, pnlPct: +27.1, delta: 0.58 },
  { symbol: 'NVDA', type: 'CALL', strike: '$480', expiry: '28 Mar', qty: 3, cost: '$12.40', price: '$18.60', pnl: +1860, pnlPct: +50.0, delta: 0.72 },
  { symbol: 'GOLD', type: 'CMDTY', strike: '-', expiry: '-', qty: 10, cost: '$2,150', price: '$2,185', pnl: +350, pnlPct: +1.6, delta: 1.00 },
];

const KPI_CARDS = [
  { label: 'Total Portfolio', value: '$48,392', sub: '+$3,240 this week', type: 'quantum', positive: true },
  { label: 'Total P&L', value: '+$8,942', sub: '+22.6% YTD', type: 'quantum', positive: true },
  { label: 'Unrealized P&L', value: '+$2,341', sub: "Today's change", type: 'nebula', positive: true },
  { label: 'Options Exposure', value: '$12,840', sub: '33.4% of portfolio', type: 'electrum', positive: null },
];

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

const PnlChart: React.FC = () => {
  const theme = useTheme();
  const points = [0, 1200, 800, 2400, 1800, 3600, 3200, 4800, 4200, 6000, 5400, 7200, 8942];
  const maxV = Math.max(...points);
  const w = 400, h = 120;
  const pts = points
    .map((v, i) => `${(i / (points.length - 1)) * w},${h - (v / maxV) * h}`)
    .join(' L ');

  return (
    <svg role="img" aria-label="Portfolio P&L Trajectory Chart" viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', height: 120, overflow: 'visible' }}>
      <defs>
        <linearGradient id="pnl-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#10b981" stopOpacity="0.4" />
          <stop offset="100%" stopColor="#10b981" stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={`M ${pts} L ${w},${h} L 0,${h} Z`} fill="url(#pnl-grad)" />
      <path d={`M ${pts}`} fill="none" stroke={theme.palette.financial.qfd.quantum} strokeWidth="2" strokeLinejoin="round" />
      {/* Benchmark */}
      <line x1={0} y1={h * 0.7} x2={w} y2={h * 0.35} stroke="#38bdf8" strokeWidth="1.5" strokeDasharray="6 4" opacity={0.5} />
    </svg>
  );
};

export const PortfolioPage: React.FC = () => {
  const theme = useTheme();

  return (
    <Container maxWidth="xl" sx={{ mt: 2, pb: 6 }}>
      {/* Header */}
      <Stack direction="row" justifyContent="space-between" alignItems="flex-end" sx={{ mb: 4 }}>
        <Box>
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
          >
            <Typography
              variant="h2"
              sx={{
                fontWeight: 900,
                mb: 0.5,
                fontFamily: 'Outfit',
                letterSpacing: '-0.04em',
                background: theme.palette.financial.qfd.iridescent,
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                textShadow: `0 0 40px ${alpha(theme.palette.financial.qfd.quantum, 0.3)}`,
              }}
            >
              Institutional Portfolio · ZENITH
            </Typography>
          </motion.div>
          <Typography variant="body2" sx={{ color: 'text.secondary', fontWeight: 500 }}>
            Mastery Verified: Multi-Asset Options & Risk Attribution Engine
          </Typography>
        </Box>
      </Stack>

      {/* KPI cards */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
      >
        <Grid container spacing={3} sx={{ mb: 4 }}>
          {KPI_CARDS.map((kpi) => {
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

      {/* Charts row */}
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

        <Grid size={{ xs: 12, lg: 4 }} className="slide-up" style={{ animationDelay: '0.2s' }}>
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
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 3 }}>
              <Stack direction="row" spacing={1.5} alignItems="center">
                <TrendingUpIcon sx={{ color: 'success.main', fontSize: 20 }} />
                <Typography variant="body1" sx={{ fontWeight: 900, fontFamily: 'Outfit' }}>
                  P&L Performance
                </Typography>
              </Stack>
            </Stack>
            <Box sx={{ borderRadius: 4, overflow: 'hidden', border: `1px solid ${alpha('#fff', 0.03)}`, p: 1 }}>
              <PnlChart />
            </Box>
          </Paper>
        </Grid>

        <Grid size={{ xs: 12, lg: 4 }} className="slide-up" style={{ animationDelay: '0.3s' }}>
          <Paper
            sx={{
              p: 1,
              height: 380,
              borderRadius: 6,
              background: `linear-gradient(135deg, ${alpha('#0f172a', 0.6)}, ${alpha('#0f172a', 0.2)})`,
              backdropFilter: 'blur(40px) saturate(200%)',
              border: `1px solid ${alpha('#f8fafc', 0.08)}`,
            }}
          >
            <PnLExplainChart data={{ delta: 4200, gamma: 1200, vega: -800, theta: -450, total: 4150 }} />
          </Paper>
        </Grid>
      </Grid>

      {/* Portfolio summary + positions */}
      <Grid container spacing={3}>
        <Grid size={{ xs: 12, lg: 4 }} className="slide-up" style={{ animationDelay: '0.1s' }}>
          <Paper sx={{ height: 440, overflow: 'hidden' }}>
            <Stack direction="row" spacing={1} alignItems="center" sx={{ p: 2, pb: 0 }}>
              <PortfolioIcon sx={{ color: 'primary.main', fontSize: 18 }} />
              <Typography variant="body2" sx={{ fontWeight: 700 }}>Summary</Typography>
            </Stack>
            <PortfolioSummary />
          </Paper>
        </Grid>

        <Grid size={{ xs: 12, lg: 8 }} className="slide-up" style={{ animationDelay: '0.2s' }}>
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
                label={`${POSITIONS.length} active units`}
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
            <Table size="small" aria-label="open-positions-table">
              <TableHead>
                <TableRow sx={{ bgcolor: alpha('#fff', 0.02) }}>
                  {['Symbol', 'Type', 'Strike', 'Expiry', 'Qty', 'Avg Cost', 'Price', 'P&L', 'P&L%', 'Δ', 'Action'].map((h) => (
                    <TableCell
                      key={h}
                      sx={{
                        color: 'text.secondary',
                        fontWeight: 900,
                        fontSize: '0.7rem',
                        letterSpacing: '0.1em',
                        borderBottom: `1px solid ${alpha('#94a3b8', 0.1)}`,
                        py: 2,
                        px: 2,
                        textTransform: 'uppercase'
                      }}
                    >
                      {h}
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {POSITIONS.map((p) => (
                  <TableRow
                    key={`${p.symbol}-${p.type}-${p.strike}`}
                    sx={{
                      '&:hover': { bgcolor: alpha('#7B68EE', 0.03) },
                      transition: 'background 0.2s'
                    }}
                  >
                    <TableCell sx={{ fontWeight: 900, py: 2, px: 2, borderBottom: `1px solid ${alpha('#94a3b8', 0.05)}`, color: 'primary.main' }}>
                      {p.symbol}
                    </TableCell>
                    <TableCell sx={{ py: 2, px: 2, borderBottom: `1px solid ${alpha('#94a3b8', 0.05)}` }}>
                      <Chip
                        label={p.type}
                        size="small"
                        sx={{
                          height: 20,
                          fontSize: '0.65rem',
                          fontWeight: 900,
                          bgcolor: alpha(
                            p.type === 'CALL' ? '#10b981' : 
                            p.type === 'PUT' ? '#f43f5e' : 
                            p.type === 'CRYPTO' ? '#fbbf24' : 
                            p.type === 'FOREX' ? '#38bdf8' : '#94a3b8', 
                            0.1
                          ),
                          color: 
                            p.type === 'CALL' ? '#10b981' : 
                            p.type === 'PUT' ? '#f43f5e' : 
                            p.type === 'CRYPTO' ? '#fbbf24' : 
                            p.type === 'FOREX' ? '#38bdf8' : '#94a3b8',
                          border: `1px solid ${alpha(
                            p.type === 'CALL' ? '#10b981' : 
                            p.type === 'PUT' ? '#f43f5e' : 
                            p.type === 'CRYPTO' ? '#fbbf24' : 
                            p.type === 'FOREX' ? '#38bdf8' : '#94a3b8', 
                            0.2
                          )}`,
                          borderRadius: 1
                        }}
                      />
                    </TableCell>
                    <TableCell sx={{ fontFamily: 'JetBrains Mono', fontWeight: 700, fontSize: '0.85rem', py: 2, px: 2, borderBottom: `1px solid ${alpha('#94a3b8', 0.05)}` }}>
                      {p.strike}
                    </TableCell>
                    <TableCell sx={{ color: 'text.secondary', fontWeight: 600, fontSize: '0.8rem', py: 2, px: 2, borderBottom: `1px solid ${alpha('#94a3b8', 0.05)}` }}>
                      {p.expiry}
                    </TableCell>
                    <TableCell sx={{ fontFamily: 'JetBrains Mono', fontWeight: 700, fontSize: '0.85rem', py: 2, px: 2, borderBottom: `1px solid ${alpha('#94a3b8', 0.05)}` }}>
                      {p.qty}
                    </TableCell>
                    <TableCell sx={{ fontFamily: 'JetBrains Mono', fontWeight: 500, fontSize: '0.85rem', color: 'text.secondary', py: 2, px: 2, borderBottom: `1px solid ${alpha('#94a3b8', 0.05)}` }}>
                      {p.cost}
                    </TableCell>
                    <TableCell sx={{ fontFamily: 'JetBrains Mono', fontWeight: 700, fontSize: '0.85rem', py: 2, px: 2, borderBottom: `1px solid ${alpha('#94a3b8', 0.05)}` }}>
                      {p.price}
                    </TableCell>
                    <TableCell
                      sx={{
                        fontWeight: 900,
                        fontFamily: 'JetBrains Mono',
                        fontSize: '0.9rem',
                        color: p.pnl > 0 ? 'success.main' : 'error.main',
                        py: 2,
                        px: 2,
                        borderBottom: `1px solid ${alpha('#94a3b8', 0.05)}`,
                      }}
                    >
                      {p.pnl > 0 ? '+' : ''}${Math.abs(p.pnl).toLocaleString()}
                    </TableCell>
                    <TableCell
                      sx={{
                        fontWeight: 900,
                        fontFamily: 'JetBrains Mono',
                        fontSize: '0.85rem',
                        color: p.pnlPct > 0 ? 'success.main' : 'error.main',
                        py: 2,
                        px: 2,
                        borderBottom: `1px solid ${alpha('#94a3b8', 0.05)}`,
                      }}
                    >
                      {p.pnlPct > 0 ? '+' : ''}{p.pnlPct}%
                    </TableCell>
                    <TableCell
                      sx={{
                        fontFamily: 'JetBrains Mono',
                        fontWeight: 700,
                        fontSize: '0.85rem',
                        color: p.delta > 0 ? '#00FFFF' : '#f43f5e',
                        py: 2,
                        px: 2,
                        borderBottom: `1px solid ${alpha('#94a3b8', 0.05)}`,
                      }}
                    >
                      {p.delta.toFixed(2)}
                    </TableCell>
                    <TableCell sx={{ py: 2, px: 2, borderBottom: `1px solid ${alpha('#94a3b8', 0.05)}` }}>
                      <Button
                        size="small"
                        variant="contained"
                        sx={{
                          py: 0.5,
                          px: 1.5,
                          fontSize: '0.65rem',
                          fontWeight: 900,
                          minWidth: 60,
                          height: 28,
                          borderRadius: 1.5,
                          bgcolor: alpha('#f43f5e', 0.1),
                          color: '#f43f5e',
                          '&:hover': { bgcolor: alpha('#f43f5e', 0.2) }
                        }}
                      >
                        LIQUIDATE
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default PortfolioPage;
