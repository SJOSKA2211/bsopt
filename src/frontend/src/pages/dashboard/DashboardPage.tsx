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
  <AnimatedCard delay={index * 0.05} className="p-6 h-full">
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
  const { isConnected } = useDataIntegration({ 
    symbols: UI_CONFIG.INSTITUTIONAL_SYMBOLS,
    enabled: true 
  });

  const systemGammaFromStore = usePricingStore((state: PricingState) => state.prices['SPX']?.gamma);
  const portfolioTotal = usePricingStore((state: PricingState) => state.portfolioTotal);
  
  const { data: systemMetrics } = useSystemMetrics();
  const { data: recentSignals, isLoading: isLoadingSignals } = useSignals(5);

  const systemGamma = systemMetrics?.gamma || systemGammaFromStore || 0;
  const vegaSens = systemMetrics?.vega ? `${(systemMetrics.vega / 1000).toFixed(2)}k` : "0.00k";
  
  return (
    <div className="p-6 bg-bento-bg">
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
           <KpiCard label="VEGA_SENS" value={vegaSens} color="#14B8A6" index={2} />
        </div>
        <div className="col-span-12 sm:col-span-6 lg:col-span-3">
           <KpiCard label="MODEL_CONFIDENCE" value="98.4%" color="#BD00FF" index={3} />
        </div>

        {/* Intelligence Layer */}
        <div className="col-span-12 lg:col-span-8">
           <AnimatedCard className="h-[520px] relative overflow-hidden group">
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-mint to-quantum opacity-20" />
              <div className="p-4 border-b border-white/5 bg-white/2 flex justify-between items-center">
                 <span className="label-secondary opacity-40 font-black tracking-widest">DEEP_INFERENCE_ENGINE // REAL_TIME_STREAM</span>
                 <div className="status-pill healthy scale-90">COMPUTING</div>
              </div>
              <div className="p-4 h-[calc(100%-60px)]">
                 <DeepInferenceEngine />
              </div>
           </AnimatedCard>
        </div>

        <div className="col-span-12 lg:col-span-4">
           <AnimatedCard className="h-[520px] !p-0 overflow-hidden relative border-white/5 hover:border-white/10 transition-all">
              <div className="absolute top-8 left-8 p-4 z-20 pointer-events-none">
                 <span className="label-secondary opacity-40">RISK_CONCENTRATION_MAP</span>
              </div>
              <RiskExposureGrid />
              <div className="absolute bottom-4 left-0 w-full p-4 bg-gradient-to-t from-black/80 to-transparent z-10">
                 <div className="flex justify-between items-center text-[10px] uppercase font-black tracking-widest text-white/40 px-4">
                    <span>HEAT_INDEX: NOMINAL</span>
                    <span>EXP: 2.1σ</span>
                 </div>
              </div>
           </AnimatedCard>
        </div>

        {/* Charts & Signals */}
        <div className="col-span-12 lg:col-span-7">
           <AnimatedCard className="h-[480px] !p-0 border-white/5 hover:border-white/10 transition-all relative group bg-white/[0.01]">
              <div className="p-6 border-b border-white/5 bg-white/2 flex justify-between items-center">
                 <span className="label-secondary opacity-40">VOLATILITY_SURFACE_v8.4</span>
                 <div className="flex gap-2">
                    <button className="px-3 py-1 bg-white/5 border border-white/10 rounded-lg text-[9px] font-black hover:bg-white/10 transition-colors">EXPORT_DATA</button>
                    <button className="px-3 py-1 bg-mint/10 border border-mint/20 rounded-lg text-[9px] font-black text-mint">REFRESH_RT</button>
                 </div>
              </div>
              <div className="p-4 h-[calc(100%-72px)] overflow-hidden">
                <Suspense fallback={<LoadingFallback />}>
                   <LivePriceChart />
                </Suspense>
              </div>
           </AnimatedCard>
        </div>

        <div className="col-span-12 lg:col-span-5">
           <AnimatedCard className="h-[480px] !p-0 border-white/5 hover:border-white/10 transition-all overflow-hidden relative">
              <div className="p-6 border-b border-white/5 bg-white/2">
                 <span className="label-secondary opacity-40">SIGNAL_PROPAGATION_LOG</span>
              </div>
              <div className="overflow-auto h-[calc(100%-72px)]">
                 {isLoadingSignals ? (
                    <LoadingFallback />
                 ) : (
                    recentSignals?.map((log, i) => (
                       <div key={i} className={`flex items-center p-5 px-8 ${i === recentSignals.length - 1 ? '' : 'border-b border-white/5'} hover:bg-white/5 transition-all group cursor-default`}>
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
