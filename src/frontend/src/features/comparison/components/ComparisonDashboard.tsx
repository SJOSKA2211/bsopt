import React, { useEffect } from 'react';
import { Box, Paper, Typography, Grid, Stack, alpha, useTheme, LinearProgress, Chip } from '@mui/material';
import { motion } from 'framer-motion';
import { useMotion } from '../../../hooks/useMotion';
import { useComparisonStore } from '../../../store/useComparisonStore';
import { Bolt as MLIconMui } from '@mui/icons-material';

// Simulate real-time metrics stream for prototype
const simulateComparisonStream = (): ReturnType<typeof setInterval> => {
    return setInterval(() => {
        useComparisonStore.getState().setMetrics({
            userPnl: 12500 + Math.random() * 500 - 250,
            aiPnl: 15200 + Math.random() * 200 - 100, // AI is slightly more stable
            userSharpe: 1.8 + Math.random() * 0.1 - 0.05,
            aiSharpe: 2.4 + Math.random() * 0.05 - 0.02,
            userWinRate: 62 + Math.random()
                * 2 - 1,
            aiWinRate: 78 + Math.random() * 1 - 0.5,
        });
    }, 2000);
}

const MetricRow = ({ label, userValue, aiValue, isPercentage = false, isCurrency = false }: any) => {
    const theme = useTheme();
    const qfd = theme.palette.financial?.qfd;
    const formatValue = (val: number) => {
        if (isCurrency) return `$${val.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
        if (isPercentage) return `${val.toFixed(1)}%`;
        return val.toFixed(2);
    };

    const aiOutperforms = aiValue > userValue;

    return (
        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1.5 }}>
            <Typography variant="body2" sx={{ color: 'text.secondary', fontWeight: 600, width: '30%' }}>{label}</Typography>
            <Box sx={{ width: '30%', textAlign: 'right' }}>
                <Typography variant="h6" sx={{ fontFamily: 'JetBrains Mono', fontWeight: 900, color: !aiOutperforms ? theme.palette.success.main : 'text.primary' }}>
                    {formatValue(userValue)}
                </Typography>
            </Box>
            <Box sx={{ width: '10%' }} display="flex" justifyContent="center">
                <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 900 }}>VS</Typography>
            </Box>
            <Box sx={{ width: '30%', textAlign: 'left' }}>
                <Typography variant="h6" sx={{ fontFamily: 'JetBrains Mono', fontWeight: 900, color: aiOutperforms ? qfd?.electrum ?? '#D4AF37' : 'text.primary' }}>
                    {formatValue(aiValue)}
                </Typography>
            </Box>
        </Stack>
    )
}

export const ComparisonDashboard: React.FC = () => {
    const theme = useTheme();
    const { variants } = useMotion();
    const metrics = useComparisonStore((state) => state.metrics);
    const modelsSelected = useComparisonStore((state) => state.modelsSelected);
    const qfd = theme.palette.financial?.qfd;

    useEffect(() => {
        // Simulate real-time metric streams
        const intervalId = simulateComparisonStream();
        return () => clearInterval(intervalId);
    }, []);

    return (
        <motion.div variants={variants.slideUp} initial="initial" animate="animate">
            <Paper
                className="qfd-glass"
                sx={{
                    p: 3,
                    borderRadius: 6,
                    border: `1px solid ${alpha('#fff', 0.05)}`,
                    position: 'relative',
                    overflow: 'hidden',
                    background: `linear-gradient(135deg, ${alpha('#0f172a', 0.6)}, ${alpha('#0f172a', 0.2)})`,
                }}
            >
                <Box
                    sx={{
                        position: 'absolute',
                        top: 0,
                        right: 0,
                        width: '100%',
                        height: 3,
                        background: `linear-gradient(90deg, transparent, ${qfd?.electrum ?? '#D4AF37'}, ${qfd?.nebula ?? '#7B68EE'})`,
                        filter: 'blur(2px)',
                    }}
                />

                <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 4 }}>
                    <Stack direction="row" spacing={1.5} alignItems="center">
                        <Box sx={{ p: 1, borderRadius: 2, bgcolor: alpha(qfd?.electrum ?? '#D4AF37', 0.1) }}>
                            <MLIconMui sx={{ color: qfd?.electrum, fontSize: 24 }} />
                        </Box>
                        <Box>
                            <Typography variant="h5" sx={{ fontWeight: 900, letterSpacing: '-0.02em', mb: 0 }}>Human vs Machine</Typography>
                            <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 600 }}>Real-time Alpha Execution Comparison</Typography>
                        </Box>
                    </Stack>
                    <Stack direction="row" spacing={1}>
                        {modelsSelected.map((m) => (
                            <Chip key={m} label={m} size="small" sx={{ bgcolor: alpha(qfd?.electrum ?? '#D4AF37', 0.1), color: qfd?.electrum, fontWeight: 900, border: `1px solid ${alpha(qfd?.electrum ?? '#D4AF37', 0.3)}` }} />
                        ))}
                    </Stack>
                </Stack>

                <Grid container spacing={4}>
                    {/* Headers */}
                    <Grid item xs={12}>
                        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ pb: 1, borderBottom: `1px solid ${alpha('#fff', 0.1)}`, mb: 1 }}>
                            <Typography variant="overline" sx={{ color: 'text.secondary', fontWeight: 900, width: '30%' }}>METRIC</Typography>
                            <Typography variant="overline" sx={{ color: 'text.secondary', fontWeight: 900, width: '30%', textAlign: 'right' }}>YOUR STRATEGY</Typography>
                            <Typography variant="overline" sx={{ width: '10%' }}></Typography>
                            <Typography variant="overline" sx={{ color: qfd?.electrum ?? '#D4AF37', fontWeight: 900, width: '30%', textAlign: 'left' }}>AI ORACLE</Typography>
                        </Stack>

                        {/* Metrics */}
                        <MetricRow label="Cumulative PnL" userValue={metrics.userPnl} aiValue={metrics.aiPnl} isCurrency />
                        <MetricRow label="Sharpe Ratio" userValue={metrics.userSharpe} aiValue={metrics.aiSharpe} />
                        <MetricRow label="Win Rate" userValue={metrics.userWinRate} aiValue={metrics.aiWinRate} isPercentage />
                    </Grid>
                </Grid>

                {/* Execution Visualizer */}
                <Box sx={{ mt: 4, pt: 3, borderTop: `1px solid ${alpha('#fff', 0.05)}` }}>
                    <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 900, mb: 1, display: 'block' }}>ALPHA SPREAD (AI DOMINANCE)</Typography>
                    <LinearProgress
                        variant="determinate"
                        value={(metrics.aiPnl / (metrics.userPnl + metrics.aiPnl)) * 100 || 50}
                        sx={{
                            height: 8,
                            borderRadius: 4,
                            bgcolor: alpha(theme.palette.success.main, 0.2),
                            '& .MuiLinearProgress-bar': {
                                bgcolor: qfd?.electrum ?? '#D4AF37',
                                borderRadius: 4,
                                boxShadow: `0 0 10px ${alpha(qfd?.electrum ?? '#D4AF37', 0.4)}`
                            }
                        }}
                    />
                    <Stack direction="row" justifyContent="space-between" sx={{ mt: 1 }}>
                        <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 600 }}>Human Alpha</Typography>
                        <Typography variant="caption" sx={{ color: qfd?.electrum ?? '#D4AF37', fontWeight: 900 }}>AI Edge: +{((metrics.aiPnl - metrics.userPnl) / metrics.userPnl * 100).toFixed(1)}%</Typography>
                    </Stack>
                </Box>
            </Paper>
        </motion.div>
    );
};
