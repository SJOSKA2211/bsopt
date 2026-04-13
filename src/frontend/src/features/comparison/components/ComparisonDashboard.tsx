import React from 'react';
import { Box, Paper, Typography, Grid, Stack, alpha, useTheme, LinearProgress, Chip } from '@mui/material';
import { motion } from 'framer-motion';
import { useMotion } from '../../../hooks/useMotion';
import { useComparisonStore } from '../../../store/useComparisonStore';
import { useComparisonData } from '../../../api/hooks';
import { Bolt as MLIconMui } from '@mui/icons-material';

interface MetricRowProps {
    label: string;
    userValue: number;
    aiValue: number;
    isPercentage?: boolean;
    isCurrency?: boolean;
}

const MetricRow = ({ label, userValue, aiValue, isPercentage = false, isCurrency = false }: MetricRowProps) => {
    const theme = useTheme();
    const financial = (theme.palette as any).financial;
    const qfd = financial?.qfd;
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
                <Typography variant="h6" sx={{ fontFamily: 'JetBrains Mono', fontWeight: 900, color: aiOutperforms ? qfd?.amber ?? '#f59e0b' : 'text.primary' }}>
                    {formatValue(aiValue)}
                </Typography>
            </Box>
        </Stack>
    )
}

export const ComparisonDashboard: React.FC = () => {
    const theme = useTheme();
    const { variants } = useMotion();
    const modelsSelected = useComparisonStore((state: any) => state.modelsSelected);
    const financial = (theme.palette as any).financial;
    const qfd = financial?.qfd;

    // Data-Driven Synchronization via Production Hook
    const { data: serverMetrics, isLoading } = useComparisonData();
    const storeMetrics = useComparisonStore((state: any) => state.metrics);
    
    // Effective metrics: hook data preferred over store (which might be legacy)
    const metrics = serverMetrics || storeMetrics;

    return (
        <motion.div variants={variants.slideUp} initial="initial" animate="animate">
            <Paper
                className="qfd-glass"
                sx={{
                    p: 4,
                    borderRadius: 8,
                    border: `1px solid ${alpha('#fff', 0.05)}`,
                    position: 'relative',
                    overflow: 'hidden',
                    background: `linear-gradient(135deg, ${alpha('#0f172a', 0.6)}, ${alpha('#0f172a', 0.2)})`,
                    backdropFilter: 'blur(30px)',
                    minHeight: 400
                }}
            >
                <Box
                    sx={{
                        position: 'absolute',
                        top: 0,
                        right: 0,
                        width: '100%',
                        height: 4,
                        background: `linear-gradient(90deg, transparent, ${qfd?.amber ?? '#f59e0b'}, ${qfd?.sky ?? '#38bdf8'})`,
                        filter: 'blur(1px)',
                    }}
                />

                <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 6 }}>
                    <Stack direction="row" spacing={2} alignItems="center">
                        <Box sx={{ p: 1.5, borderRadius: 3, bgcolor: alpha(qfd?.amber ?? '#f59e0b', 0.1), border: `1px solid ${alpha(qfd?.amber ?? '#f59e0b', 0.2)}` }}>
                            <MLIconMui sx={{ color: qfd?.amber, fontSize: 28 }} aria-label="Alpha Comparison Icon" />
                        </Box>
                        <Box>
                            <Typography variant="h5" sx={{ fontWeight: 900, fontFamily: 'Outfit', letterSpacing: '-0.02em', mb: 0.5 }}>Human vs Machine</Typography>
                            <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 700, letterSpacing: '0.05em' }}>QUANTUM ALPHA EXECUTION MANIFOLD</Typography>
                        </Box>
                    </Stack>
                    <Stack direction="row" spacing={1}>
                        {modelsSelected.map((m: string) => (
                            <Chip key={m} label={m} size="small" sx={{ bgcolor: alpha(qfd?.amber ?? '#f59e0b', 0.1), color: qfd?.amber, fontWeight: 900, border: `1px solid ${alpha(qfd?.amber ?? '#f59e0b', 0.3)}`, fontFamily: 'JetBrains Mono', fontSize: '0.65rem' }} />
                        ))}
                    </Stack>
                </Stack>

                <Grid container spacing={4}>
                    <Grid size={{ xs: 12 }}>
                        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ pb: 1.5, borderBottom: `1px solid ${alpha('#fff', 0.05)}`, mb: 2 }}>
                            <Typography variant="overline" sx={{ color: 'text.secondary', fontWeight: 900, width: '30%', letterSpacing: '0.1em' }}>METRIC</Typography>
                            <Typography variant="overline" sx={{ color: 'text.secondary', fontWeight: 900, width: '30%', textAlign: 'right', letterSpacing: '0.1em' }}>YOUR STRATEGY</Typography>
                            <Typography variant="overline" sx={{ width: '10%' }}></Typography>
                            <Typography variant="overline" sx={{ color: qfd?.amber ?? '#f59e0b', fontWeight: 900, width: '30%', textAlign: 'left', letterSpacing: '0.1em' }}>AI ORACLE</Typography>
                        </Stack>

                        {isLoading && !metrics ? (
                             <Box sx={{ py: 4, textAlign: 'center' }}>
                                 <LinearProgress sx={{ borderRadius: 4, height: 2, bgcolor: alpha('#fff', 0.05) }} />
                                 <Typography variant="caption" sx={{ mt: 2, display: 'block', color: 'text.secondary', fontWeight: 800 }}>SYNCHRONIZING ORACLE FEED...</Typography>
                             </Box>
                        ) : (
                            <>
                                <MetricRow label="Cumulative PnL" userValue={metrics.userPnl} aiValue={metrics.aiPnl} isCurrency />
                                <MetricRow label="Sharpe Ratio" userValue={metrics.userSharpe} aiValue={metrics.aiSharpe} />
                                <MetricRow label="Win Rate" userValue={metrics.userWinRate} aiValue={metrics.aiWinRate} isPercentage />
                                
                                <Box sx={{ mt: 6, pt: 4, borderTop: `1px solid ${alpha('#fff', 0.05)}` }}>
                                    <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
                                        <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 900, letterSpacing: '0.15em' }}>ALPHA SPREAD (AI DOMINANCE)</Typography>
                                        <Typography variant="caption" sx={{ color: qfd?.amber ?? '#f59e0b', fontWeight: 900, fontFamily: 'JetBrains Mono' }}>
                                            AI EDGE: +{((metrics.aiPnl - metrics.userPnl) / (metrics.userPnl || 1) * 100).toFixed(1)}%
                                        </Typography>
                                    </Stack>
                                    <LinearProgress
                                        variant="determinate"
                                        value={(metrics.aiPnl / ((metrics.userPnl + metrics.aiPnl) || 1)) * 100}
                                        sx={{
                                            height: 10,
                                            borderRadius: 5,
                                            bgcolor: alpha(theme.palette.success.main, 0.1),
                                            '& .MuiLinearProgress-bar': {
                                                bgcolor: qfd?.amber ?? '#f59e0b',
                                                borderRadius: 5,
                                                boxShadow: `0 0 15px ${alpha(qfd?.amber ?? '#f59e0b', 0.4)}`
                                            }
                                        }}
                                    />
                                    <Stack direction="row" justifyContent="space-between" sx={{ mt: 1.5 }}>
                                        <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 800 }}>HUMAN RETAIL</Typography>
                                        <Typography variant="caption" sx={{ color: qfd?.amber ?? '#f59e0b', fontWeight: 800 }}>ORACLE CLUSTER</Typography>
                                    </Stack>
                                </Box>
                            </>
                        )}
                    </Grid>
                </Grid>
            </Paper>
        </motion.div>
    );
};
