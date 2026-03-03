import React, { lazy, Suspense } from 'react';
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
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Button,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  AccountBalance as PortfolioIcon,
  Assessment as AllocationIcon,
} from '@mui/icons-material';
import { PortfolioSummary } from '../../features/portfolio/components/PortfolioSummary';
import { PositionsSummary } from '../../features/portfolio/components/PositionsSummary';

const LoadingFallback = () => (
  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', p: 4 }}>
    <CircularProgress size={28} />
  </Box>
);

// ---- Mock open positions ----
const POSITIONS = [
  { symbol: 'AAPL', type: 'CALL', strike: '$185', expiry: '21 Mar', qty: 5, cost: '$4.20', price: '$6.88', pnl: +1340, pnlPct: +63.8, delta: 0.64 },
  { symbol: 'AAPL', type: 'PUT', strike: '$190', expiry: '21 Mar', qty: 3, cost: '$3.80', price: '$2.95', pnl: -255, pnlPct: -22.4, delta: -0.41 },
  { symbol: 'SPY', type: 'CALL', strike: '$470', expiry: '18 Apr', qty: 10, cost: '$5.60', price: '$7.12', pnl: +1520, pnlPct: +27.1, delta: 0.58 },
  { symbol: 'QQQ', type: 'CALL', strike: '$400', expiry: '18 Apr', qty: 8, cost: '$6.10', price: '$8.04', pnl: +1552, pnlPct: +31.8, delta: 0.61 },
  { symbol: 'NVDA', type: 'CALL', strike: '$480', expiry: '28 Mar', qty: 3, cost: '$12.40', price: '$18.60', pnl: +1860, pnlPct: +50.0, delta: 0.72 },
  { symbol: 'TSLA', type: 'PUT', strike: '$250', expiry: '18 Apr', qty: 6, cost: '$8.20', price: '$7.45', pnl: -450, pnlPct: -9.1, delta: -0.38 },
];

const KPI_CARDS = [
  { label: 'Total Portfolio', value: '$48,392', sub: '+$3,240 this week', positive: true, color: '#10b981' },
  { label: 'Total P&L', value: '+$8,942', sub: '+22.6% YTD', positive: true, color: '#10b981' },
  { label: 'Unrealized P&L', value: '+$2,341', sub: 'Today\'s change', positive: true, color: '#38bdf8' },
  { label: 'Options Exposure', value: '$12,840', sub: '33.4% of portfolio', positive: null, color: '#fbbf24' },
];

// Simple SVG donut chart
const DonutChart: React.FC = () => {
  const segments = [
    { label: 'AAPL', pct: 28, color: '#10b981' },
    { label: 'SPY', pct: 22, color: '#38bdf8' },
    { label: 'QQQ', pct: 18, color: '#a855f7' },
    { label: 'NVDA', pct: 15, color: '#fbbf24' },
    { label: 'Cash', pct: 17, color: '#64748b' },
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
      <svg viewBox="0 0 180 180" style={{ width: '100%', maxWidth: 180 }}>
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

// Simple SVG P&L line chart
const PnlChart: React.FC = () => {
  const points = [0, 1200, 800, 2400, 1800, 3600, 3200, 4800, 4200, 6000, 5400, 7200, 8942];
  const maxV = Math.max(...points);
  const w = 400, h = 120;
  const pts = points
    .map((v, i) => `${(i / (points.length - 1)) * w},${h - (v / maxV) * h}`)
    .join(' L ');

  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', height: 120, overflow: 'visible' }}>
      <defs>
        <linearGradient id="pnl-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#10b981" stopOpacity="0.4" />
          <stop offset="100%" stopColor="#10b981" stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={`M ${pts} L ${w},${h} L 0,${h} Z`} fill="url(#pnl-grad)" />
      <path d={`M ${pts}`} fill="none" stroke="#10b981" strokeWidth="2" strokeLinejoin="round" />
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
          <Typography variant="h3" className="text-gradient slide-up" sx={{ fontWeight: 800, mb: 0.5 }}>
            Portfolio Returns &amp; Holdings
          </Typography>
          <Typography variant="body2" sx={{ color: 'text.disabled' }}>
            Options positions, P&L analytics &amp; allocation overview
          </Typography>
        </Box>
      </Stack>

      {/* KPI cards */}
      <Grid container spacing={2.5} sx={{ mb: 4 }}>
        {KPI_CARDS.map((kpi) => (
          <Grid key={kpi.label} size={{ xs: 12, sm: 6, lg: 3 }}>
            <Paper
              className="stat-card"
              sx={{ p: 2.5, border: `1px solid ${alpha(kpi.color, 0.15)}`, height: '100%' }}
            >
              <Box sx={{ position: 'absolute', top: -20, right: -20, width: 80, height: 80, borderRadius: '50%', bgcolor: alpha(kpi.color, 0.08), filter: 'blur(20px)', pointerEvents: 'none' }} />
              <Typography variant="caption" sx={{ color: 'text.disabled', fontWeight: 700, letterSpacing: '0.1em' }}>
                {kpi.label}
              </Typography>
              <Typography
                variant="h4"
                sx={{ fontWeight: 800, my: 0.75, fontFamily: '"JetBrains Mono", monospace', color: kpi.positive ? kpi.color : 'text.primary', fontSize: '1.4rem' }}
              >
                {kpi.value}
              </Typography>
              <Typography variant="caption" sx={{ color: kpi.positive ? kpi.color : kpi.positive === false ? 'error.main' : 'warning.main', fontWeight: 600 }}>
                {kpi.sub}
              </Typography>
            </Paper>
          </Grid>
        ))}
      </Grid>

      {/* Charts row */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {/* Donut allocation */}
        <Grid size={{ xs: 12, lg: 4 }} className="slide-up" style={{ animationDelay: '0.1s' }}>
          <Paper sx={{ p: 3, height: 380 }}>
            <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
              <AllocationIcon sx={{ color: 'secondary.main', fontSize: 18 }} />
              <Typography variant="body1" sx={{ fontWeight: 700 }}>
                Allocation
              </Typography>
            </Stack>
            <DonutChart />
          </Paper>
        </Grid>

        {/* P&L performance line chart */}
        <Grid size={{ xs: 12, lg: 8 }} className="slide-up" style={{ animationDelay: '0.2s' }}>
          <Paper sx={{ p: 3, height: 380 }}>
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
              <Stack direction="row" spacing={1} alignItems="center">
                <TrendingUpIcon sx={{ color: 'success.main', fontSize: 18 }} />
                <Typography variant="body1" sx={{ fontWeight: 700 }}>
                  P&L Performance · YTD
                </Typography>
              </Stack>
              <Stack direction="row" spacing={2}>
                <Stack direction="row" spacing={0.75} alignItems="center">
                  <Box sx={{ width: 20, height: 2, bgcolor: 'success.main', borderRadius: 1 }} />
                  <Typography variant="caption" sx={{ color: 'text.disabled' }}>Portfolio</Typography>
                </Stack>
                <Stack direction="row" spacing={0.75} alignItems="center">
                  <Box sx={{ width: 20, height: 2, bgcolor: 'secondary.main', borderRadius: 1, opacity: 0.5 }} />
                  <Typography variant="caption" sx={{ color: 'text.disabled' }}>Benchmark</Typography>
                </Stack>
              </Stack>
            </Stack>
            <Box>
              <PnlChart />
            </Box>
            <Stack direction="row" justifyContent="space-between" sx={{ mt: 1, px: 0.5 }}>
              {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Now'].map((m) => (
                <Typography key={m} variant="caption" sx={{ color: 'text.disabled', fontSize: '0.6rem' }}>
                  {m}
                </Typography>
              ))}
            </Stack>
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
          <Paper sx={{ overflow: 'hidden' }}>
            <Stack direction="row" spacing={1.5} alignItems="center" sx={{ p: 2.5, pb: 1.5 }}>
              <Typography variant="h5" sx={{ fontWeight: 700 }}>Open Positions</Typography>
              <Chip
                label={`${POSITIONS.length} positions`}
                size="small"
                sx={{ height: 22, fontSize: '0.68rem', fontWeight: 600, bgcolor: alpha('#94a3b8', 0.1), color: 'text.secondary' }}
              />
            </Stack>
            <Table size="small" aria-label="open-positions-table">
              <TableHead>
                <TableRow>
                  {['Symbol', 'Type', 'Strike', 'Expiry', 'Qty', 'Avg Cost', 'Price', 'P&L', 'P&L%', 'Δ', 'Action'].map((h) => (
                    <TableCell
                      key={h}
                      sx={{
                        color: 'text.disabled',
                        fontWeight: 700,
                        fontSize: '0.68rem',
                        letterSpacing: '0.07em',
                        borderColor: alpha('#94a3b8', 0.08),
                        py: 1,
                        px: 1.5,
                      }}
                    >
                      {h}
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {POSITIONS.map((p) => (
                  <TableRow key={`${p.symbol}-${p.type}-${p.strike}`}>
                    <TableCell sx={{ fontWeight: 700, py: 1.25, px: 1.5, borderColor: alpha('#94a3b8', 0.06) }}>
                      {p.symbol}
                    </TableCell>
                    <TableCell sx={{ py: 1.25, px: 1.5, borderColor: alpha('#94a3b8', 0.06) }}>
                      <Chip
                        label={p.type}
                        size="small"
                        color={p.type === 'CALL' ? 'success' : 'error'}
                        sx={{ height: 18, fontSize: '0.6rem', fontWeight: 700 }}
                      />
                    </TableCell>
                    <TableCell sx={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '0.8rem', py: 1.25, px: 1.5, borderColor: alpha('#94a3b8', 0.06) }}>
                      {p.strike}
                    </TableCell>
                    <TableCell sx={{ color: 'text.disabled', fontSize: '0.78rem', py: 1.25, px: 1.5, borderColor: alpha('#94a3b8', 0.06) }}>
                      {p.expiry}
                    </TableCell>
                    <TableCell sx={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '0.8rem', py: 1.25, px: 1.5, borderColor: alpha('#94a3b8', 0.06) }}>
                      {p.qty}
                    </TableCell>
                    <TableCell sx={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '0.8rem', color: 'text.secondary', py: 1.25, px: 1.5, borderColor: alpha('#94a3b8', 0.06) }}>
                      {p.cost}
                    </TableCell>
                    <TableCell sx={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '0.8rem', py: 1.25, px: 1.5, borderColor: alpha('#94a3b8', 0.06) }}>
                      {p.price}
                    </TableCell>
                    <TableCell
                      sx={{
                        fontWeight: 700,
                        fontFamily: '"JetBrains Mono", monospace',
                        fontSize: '0.82rem',
                        color: p.pnl > 0 ? 'success.main' : 'error.main',
                        py: 1.25,
                        px: 1.5,
                        borderColor: alpha('#94a3b8', 0.06),
                      }}
                    >
                      {p.pnl > 0 ? '+' : ''}${Math.abs(p.pnl)}
                    </TableCell>
                    <TableCell
                      sx={{
                        fontWeight: 700,
                        fontFamily: '"JetBrains Mono", monospace',
                        fontSize: '0.78rem',
                        color: p.pnlPct > 0 ? 'success.main' : 'error.main',
                        py: 1.25,
                        px: 1.5,
                        borderColor: alpha('#94a3b8', 0.06),
                      }}
                    >
                      {p.pnlPct > 0 ? '+' : ''}{p.pnlPct}%
                    </TableCell>
                    <TableCell
                      sx={{
                        fontFamily: '"JetBrains Mono", monospace',
                        fontSize: '0.78rem',
                        color: p.delta > 0 ? 'info.main' : 'error.main',
                        py: 1.25,
                        px: 1.5,
                        borderColor: alpha('#94a3b8', 0.06),
                      }}
                    >
                      {p.delta.toFixed(2)}
                    </TableCell>
                    <TableCell sx={{ py: 1.25, px: 1.5, borderColor: alpha('#94a3b8', 0.06) }}>
                      <Button
                        size="small"
                        variant="outlined"
                        color="error"
                        sx={{
                          py: 0.25,
                          px: 1,
                          fontSize: '0.65rem',
                          fontWeight: 700,
                          minWidth: 48,
                          height: 24,
                          borderRadius: 1.5,
                        }}
                      >
                        Close
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
