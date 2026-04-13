import React, { lazy, Suspense } from 'react';
import { DeepInferenceEngine } from '../../features/dashboard/components/DeepInferenceEngine';
import { RiskExposureGrid } from '../../features/dashboard/components/RiskExposureGrid';
import { motion } from 'framer-motion';
import { AnimatedCard } from '../../components/common/AnimatedCard';
import { usePricingStore, type PricingState } from '../../store/usePricingStore';
import { useDataIntegration } from '../../hooks/useDataIntegration';
import { useSignals } from '../../api/hooks';
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
  const systemGamma = usePricingStore((state: PricingState) => state.systemGamma);
  const portfolioTotal = usePricingStore((state: PricingState) => state.portfolioTotal);
  
  const { data: recentSignals, isLoading: isLoadingSignals } = useSignals(5);
  
  return (
    <div className="p-6 h-[calc(100vh-64px)] overflow-auto bg-bento-bg">
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="bento-grid"
      >
        {/* KPI Row */}
        <div className="col-span-12 sm:col-span-6 lg:col-span-3">
           <KpiCard label="SYSTEM_GAMMA" value={systemGamma.toFixed(3)} color="#00FFA3" index={0} />
        </div>
        <div className="col-span-12 sm:col-span-6 lg:col-span-3">
           <KpiCard label="PORTFOLIO_NAV" value={portfolioTotal} color="#F59E0B" prefix="$" index={1} />
        </div>
        <div className="col-span-12 sm:col-span-6 lg:col-span-3">
           <KpiCard label="VEGA_SENS" value="4.52k" color="#14B8A6" index={2} />
        </div>
        <div className="col-span-12 sm:col-span-6 lg:col-span-3">
           <KpiCard label="WS_STATUS" value={isConnected ? 'CONNECTED' : 'CONNECTING...'} color={isConnected ? '#00FFA3' : '#F59E0B'} index={3} />
        </div>

        {/* Intelligence Layer */}
        <div className="col-span-12 lg:col-span-4">
           <AnimatedCard delay={0.2} className="h-full">
              <h2 className="label-secondary mb-6 opacity-40">DEEP_INFERENCE_ENGINE // v4.2</h2>
              <DeepInferenceEngine symbol="SPX" />
           </AnimatedCard>
        </div>
        
        <div className="col-span-12 lg:col-span-4">
           <AnimatedCard delay={0.25} className="h-full">
              <h2 className="label-secondary mb-6 opacity-40">RISK_EXPOSURE_GRID</h2>
              <RiskExposureGrid />
           </AnimatedCard>
        </div>

        <div className="col-span-12 lg:col-span-4">
           <AnimatedCard delay={0.3} className="h-full">
              <h2 className="label-secondary mb-6 opacity-40">STRATEGY_ALLOCATION</h2>
              <div className="flex flex-col gap-6">
                 {UI_CONFIG.STRATEGY_ALLOCATIONS.map(strat => (
                   <div key={strat.name}>
                      <div className="flex justify-between mb-2">
                         <span className="text-[11px] font-bold text-white/80">{strat.name}</span>
                         <span className="data-mono text-[11px] font-bold" style={{ color: strat.color }}>{strat.weight}%</span>
                      </div>
                      <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
                         <motion.div 
                           initial={{ width: 0 }}
                           animate={{ width: `${strat.weight}%` }}
                           transition={{ duration: 1, delay: 0.5 }}
                           className="h-full"
                           style={{ backgroundColor: strat.color }} 
                         />
                      </div>
                   </div>
                 ))}
              </div>
           </AnimatedCard>
        </div>

        {/* Observation Deck */}
        <div className="col-span-12">
           <AnimatedCard delay={0.4} className="h-[540px] !p-0">
              <div className="p-6 border-b border-bento-border flex justify-between items-center">
                 <h2 className="label-secondary opacity-40">TEMPORAL_TRAJECTORY // GLOBAL_INDICES</h2>
                 <div className="flex gap-2">
                    {UI_CONFIG.TIME_FRAMES.map(tf => (
                      <button key={tf} className={`px-3 py-1 rounded text-[10px] font-bold border transition-colors ${tf === UI_CONFIG.DEFAULT_TIME_FRAME ? 'bg-mint text-black border-mint' : 'bg-white/5 text-white/40 border-white/5 hover:bg-white/10'}`}>
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

        {/* Signals Telemetry */}
        <div className="col-span-12">
           <AnimatedCard delay={0.5} className="!p-0">
              <div className="p-6 border-b border-bento-border flex justify-between items-center">
                 <h2 className="label-secondary opacity-40">SIGNAL_TELEMETRY</h2>
                 <div className={`status-pill ${isConnected ? 'healthy bg-mint/10 border-mint/20 text-mint' : 'bg-amber-500/10 border-amber-500/20 text-amber-500'}`}>
                   {isConnected ? 'LIVE_FEED' : 'CONNECT_ERROR'}
                 </div>
              </div>
              <div className="p-2 min-h-[300px]">
                 {isLoadingSignals ? (
                   <div className="flex justify-center p-12">
                      <div className="w-5 h-5 border-2 border-mint/20 border-t-mint rounded-full animate-spin" />
                   </div>
                 ) : recentSignals?.data?.length === 0 ? (
                    <div className="p-8 text-center text-white/30 text-xs">NO_SIGNALS_DETECTED</div>
                 ) : (
                    recentSignals?.data?.map((log: any, i: number) => (
                      <div key={i} className={`flex items-center p-4 px-6 ${i === recentSignals.data.length - 1 ? '' : 'border-b border-white/5'} hover:bg-white/5 transition-colors group cursor-default text-white/90`}>
                         <span className="data-mono text-[11px] text-white/40 w-24 group-hover:text-mint transition-colors">
                           {new Date(log.timestamp).toLocaleTimeString()}
                         </span>
                         <div className={`status-pill mr-6 font-black scale-90 ${log.type === 'ML' ? 'bg-purple-500/10 text-purple-400' : 'bg-mint/10 text-mint'}`}>
                            {log.type}
                         </div>
                         <span className="text-sm font-medium">{log.message}</span>
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
