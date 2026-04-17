import React, { lazy, Suspense } from 'react';
import { DeepInferenceEngine } from '../../features/dashboard/components/DeepInferenceEngine';
import { RiskExposureGrid } from '../../features/dashboard/components/RiskExposureGrid';
import { motion } from 'framer-motion';
import { AnimatedCard } from '../../components/common/AnimatedCard';
import { usePricingStore, type PricingState } from '../../store/usePricingStore';
import { useDataIntegration } from '../../hooks/useDataIntegration';
import { useSignals, useSystemMetrics } from '../../api/hooks';
import { UI_CONFIG } from '../../lib/config';

// Lazy loaded components
const LivePriceChart = lazy(() =>
  import('../../features/charts/components/LivePriceChart').then(m => ({ default: m.LivePriceChart }))
);

const LoadingFallback = () => (
  <div className="flex justify-center items-center p-8">
    <div className="w-5 h-5 border-2 border-mint/20 border-t-mint rounded-full animate-spin" />
  </div>
);

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

const KpiCard: React.FC<{ label: string; value: string | number; color: string; prefix?: string; index: number }> = ({ label, value, color, prefix, index }) => (
  <AnimatedCard delay={index * 0.05} className="p-6">
     <div className="flex flex-col gap-1">
        <span className="label-secondary text-[11px] opacity-60 uppercase tracking-tighter">{label}</span>
        <div className="flex items-baseline gap-1">
           <span className="data-mono text-3xl font-extrabold text-white">
             {prefix}{typeof value === 'number' ? value.toLocaleString() : value}
           </span>
        </div>
        <div className="h-0.5 w-10 mt-2 rounded-full shadow-[0_0_8px_currentcolor]" style={{ backgroundColor: color, color }} />
     </div>
  </AnimatedCard>
);

export const DashboardPage: React.FC = () => {
  // Bootstrap data integration for institutional symbols
  const { isConnected } = useDataIntegration({ 
    symbols: UI_CONFIG.INSTITUTIONAL_SYMBOLS,
    enabled: true 
  });

  // Performance-optimized store selectors
  const systemGammaFromStore = usePricingStore((state: PricingState) => state.prices['SPX']?.gamma); // Fixed incorrect selector
  const portfolioTotal = usePricingStore((state: PricingState) => state.portfolioTotal);
  
  const { data: systemMetrics } = useSystemMetrics();
  const { data: recentSignals, isLoading: isLoadingSignals } = useSignals(5);

  const systemGamma = systemMetrics?.gamma || systemGammaFromStore || 0;
  const vegaSens = systemMetrics?.vega ? `${(systemMetrics.vega / 1000).toFixed(2)}k` : "0.00k";
  
  return (
    <div className="p-6 h-full overflow-auto bg-bento-bg">
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="bento-grid"
      >
        {/* KPI Row - Fixed 12-Column Layout */}
        <div className="col-span-12 sm:col-span-6 lg:col-span-3">
           <KpiCard label="SYSTEM_GAMMA" value={systemGamma.toFixed(3)} color="#00FFA3" index={0} />
        </div>
        <div className="col-span-12 sm:col-span-6 lg:col-span-3">
           <KpiCard label="PORTFOLIO_NAV" value={portfolioTotal} color="#F59E0B" prefix="$" index={1} />
        </div>
        <div className="col-span-12 sm:col-span-6 lg:col-span-3">
           <KpiCard label="VEGA_SENS" value={vegaSens} color="#14B8A6" index={2} />
        </div>
        <div className="col-span-12 sm:col-span-6 lg:col-span-3">
           <KpiCard label="WS_STATUS" value={isConnected ? 'CONNECTED' : 'CONNECTING...'} color={isConnected ? '#00FFA3' : '#F59E0B'} index={3} />
        </div>

        {/* Intelligence Layer - Secondary Row */}
        <div className="col-span-12 lg:col-span-4">
           <AnimatedCard delay={0.2} className="h-full min-h-[400px]">
              <h2 className="label-secondary mb-6 opacity-40">DEEP_INFERENCE_ENGINE // v4.2</h2>
              <DeepInferenceEngine symbol="SPX" />
           </AnimatedCard>
        </div>
        
        <div className="col-span-12 lg:col-span-4">
           <AnimatedCard delay={0.25} className="h-full min-h-[400px]">
              <h2 className="label-secondary mb-6 opacity-40">RISK_EXPOSURE_GRID</h2>
              <RiskExposureGrid />
           </AnimatedCard>
        </div>

        <div className="col-span-12 lg:col-span-4">
           <AnimatedCard delay={0.3} className="h-full min-h-[400px]">
              <h2 className="label-secondary mb-6 opacity-40">STRATEGY_ALLOCATION</h2>
              <div className="flex flex-col gap-8 mt-4">
                 {UI_CONFIG.STRATEGY_ALLOCATIONS.map(strat => (
                   <div key={strat.name}>
                      <div className="flex justify-between mb-3">
                         <span className="text-[11px] font-bold text-white/80 tracking-wide">{strat.name}</span>
                         <span className="data-mono text-[11px] font-bold" style={{ color: strat.color }}>{strat.weight}%</span>
                      </div>
                      <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                         <motion.div 
                           initial={{ width: 0 }}
                           animate={{ width: `${strat.weight}%` }}
                           transition={{ duration: 1, delay: 0.5 }}
                           className="h-full shadow-[0_0_10px_currentcolor]"
                           style={{ backgroundColor: strat.color, color: strat.color }} 
                         />
                      </div>
                   </div>
                 ))}
              </div>
           </AnimatedCard>
        </div>

        {/* Observation Deck - Primary Data Viz */}
        <div className="col-span-12">
           <AnimatedCard delay={0.4} className="h-[600px] !p-0 overflow-hidden">
              <div className="p-6 border-b border-bento-border flex justify-between items-center bg-white/[0.02]">
                 <h2 className="label-secondary opacity-40">TEMPORAL_TRAJECTORY // GLOBAL_INDICES</h2>
                 <div className="flex gap-2">
                    {UI_CONFIG.TIME_FRAMES.map(tf => (
                      <button key={tf} className={`px-4 py-1.5 rounded-lg text-[10px] font-black border transition-all ${tf === UI_CONFIG.DEFAULT_TIME_FRAME ? 'bg-mint text-black border-mint shadow-[0_0_15px_#00FFA3]' : 'bg-white/5 text-white/40 border-white/5 hover:bg-white/10'}`}>
                        {tf}
                      </button>
                    ))}
                 </div>
              </div>
              <div className="p-4 h-[calc(100%-73px)]">
                 <Suspense fallback={<LoadingFallback />}>
                    <LivePriceChart symbol="SPX" />
                 </Suspense>
              </div>
           </AnimatedCard>
        </div>

        {/* Signals Telemetry - System Logs */}
        <div className="col-span-12 lg:col-span-12">
           <AnimatedCard delay={0.5} className="!p-0 border-mint/20">
              <div className="p-6 border-b border-bento-border flex justify-between items-center bg-white/[0.02]">
                 <h2 className="label-secondary opacity-40 tracking-[0.2em]">SIGNAL_TELEMETRY // LIVE_HEDGE_SUBSYSTEM</h2>
                 <div className={`status-pill ${isConnected ? 'healthy' : 'warning animate-pulse'}`}>
                    {isConnected ? 'LIVE_STREAM_ACTIVE' : 'FEED_INITIALIZING'}
                 </div>
              </div>
              <div className="p-2 min-h-[320px]">
                 {isLoadingSignals ? (
                   <div className="flex flex-col items-center justify-center p-20 gap-4">
                      <div className="w-8 h-8 border-2 border-mint/10 border-t-mint rounded-full animate-spin" />
                      <span className="label-secondary opacity-20 text-[9px]">Awaiting Uplink...</span>
                   </div>
                 ) : recentSignals?.data?.length === 0 ? (
                    <div className="p-20 text-center text-white/20 text-[10px] font-black tracking-widest italic"> [ NO_SIGNALS_DETECTED_IN_CURRENT_EPOCH ] </div>
                 ) : (
                    recentSignals?.data?.map((log: any, i: number) => (
                      <div key={i} className={`flex items-center p-5 px-8 ${i === recentSignals.data.length - 1 ? '' : 'border-b border-white/5'} hover:bg-white/5 transition-all group cursor-default`}>
                         <span className="data-mono text-[11px] text-white/30 w-32 group-hover:text-mint transition-colors shrink-0">
                           {new Date(log.timestamp).toLocaleTimeString()}
                         </span>
                         <div className={`status-pill mr-10 font-black scale-90 shrink-0 ${log.type === 'ML' ? 'bg-purple-500/10 text-purple-400' : 'bg-mint/10 text-mint'}`}>
                            {log.type}
                         </div>
                         <span className="text-[13px] font-medium text-white/80 group-hover:text-white transition-colors tracking-tight">{log.message}</span>
                         <div className="ml-auto opacity-0 group-hover:opacity-100 transition-opacity">
                            <span className="text-[10px] font-black text-mint/40 tracking-widest">DETAILS -&gt;</span>
                         </div>
                      </div>
                    ))
                 )}
              </div>
           </AnimatedCard>
        </div>
      </motion.div>
    </div>
  );
};

export default DashboardPage;
