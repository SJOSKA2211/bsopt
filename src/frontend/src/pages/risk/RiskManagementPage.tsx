import React from 'react';
import { PortfolioHealth } from '../../features/risk/components/PortfolioHealth';
import { BlackSwanStressTest } from '../../features/risk/components/BlackSwanStressTest';
import { motion } from 'framer-motion';
import { AnimatedCard } from '../../components/common/AnimatedCard';

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

const RiskManagementPage: React.FC = () => {
  return (
    <div className="p-6 h-full overflow-auto bg-bento-bg">
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="bento-grid"
      >
        {/* Header Section */}
        <div className="col-span-12">
           <PortfolioHealth />
        </div>

        {/* Risk Analysis Grid */}
        <div className="col-span-12 lg:col-span-8">
           <AnimatedCard className="h-[520px] !p-0 border-white/5 hover:border-white/10 transition-all overflow-hidden relative group bg-white/[0.01]">
              <div className="absolute inset-0 bg-bento-grid bg-[length:32px_32px] opacity-[0.03] group-hover:opacity-[0.05] transition-opacity" />
              <div className="p-6 border-b border-white/5 bg-white/2 flex justify-between items-center">
                 <span className="label-secondary opacity-40 font-black tracking-widest">QUANT_RISK_MANIFOLD // P&L_SURFACE_MONITOR</span>
                 <div className="status-pill healthy scale-90">COMPUTING_RT</div>
              </div>
              <div className="p-4 h-[calc(100%-72px)] flex items-center justify-center relative">
                 <div className="w-full h-full border border-white/5 bg-black/20 relative overflow-hidden flex items-center justify-center rounded-xl">
                    <div className="absolute top-[20%] left-[30%] w-64 h-64 bg-mint/5 blur-[100px] pointer-events-none" />
                    <div className="relative z-10 text-center">
                       <span className="label-secondary opacity-20 text-[10px] font-black tracking-[0.3em]">
                          [ HD_RISK_HEATMAP_INITIALIZING... ]<br/>
                          GPU_ACCELERATED_PARALLEL_SCAN: ACTIVE
                       </span>
                    </div>
                 </div>
              </div>
           </AnimatedCard>
        </div>

        <div className="col-span-12 lg:col-span-4">
           <BlackSwanStressTest />
        </div>

        {/* Detailed Metrics */}
        <div className="col-span-12 md:col-span-6">
           <AnimatedCard className="!p-0 border-white/5 hover:border-white/10 transition-all overflow-hidden relative min-h-[400px]">
              <div className="absolute top-0 left-8 px-4 py-1.5 bg-amber text-black text-[10px] font-black tracking-widest skew-x-[-15deg] z-10 shadow-[0_0_15px_rgba(255,170,0,0.3)]">
                 HEDGE_SUB_ORCHESTRATOR
              </div>
              <div className="p-6 pt-10 border-b border-white/5 bg-white/2">
                 <span className="label-secondary opacity-40">ACTIVE_DERIVATIVE_PROTECTION</span>
              </div>
              <div className="p-2">
                 {[
                   { symbol: 'SPY_241220_P_450', delta: '-124.5', type: 'PUT', status: 'ACTIVE' },
                   { symbol: 'VIX_250117_C_25', delta: '+45.2', type: 'CALL', status: 'WAITING' },
                   { symbol: 'TLT_250321_L_95', delta: '+12.0', type: 'EQUITY', status: 'ACTIVE' },
                 ].map((hedge, i) => (
                   <div key={i} className={`flex items-center p-5 ${i === 2 ? '' : 'border-b border-white/5'} hover:bg-white/5 transition-all cursor-default`}>
                      <div className="flex-1">
                         <div className="text-[11px] font-black text-white tracking-widest">{hedge.symbol}</div>
                         <div className="label-secondary text-[9px] opacity-40 mt-1">{hedge.type} // STATUS: {hedge.status}</div>
                      </div>
                      <div className="flex-1 text-right">
                         <div className={`data-mono text-[12px] font-black ${hedge.delta.includes('-') ? 'text-red-500' : 'text-mint'}`}>
                            {hedge.delta} Δ
                         </div>
                         <div className="h-1 w-12 ml-auto mt-2 bg-white/10 rounded-full overflow-hidden">
                            <div className={`h-full w-[70%] ${hedge.delta.includes('-') ? 'bg-red-500' : 'bg-mint'}`} />
                         </div>
                      </div>
                   </div>
                 ))}
              </div>
           </AnimatedCard>
        </div>

        <div className="col-span-12 md:col-span-6">
           <AnimatedCard className="!p-0 border-white/5 hover:border-white/10 transition-all overflow-hidden relative min-h-[400px]">
              <div className="p-6 border-b border-white/5 bg-white/2">
                 <span className="label-secondary opacity-40">MARGIN_LATENCY_MONITOR // CROSS_COLLATERAL</span>
              </div>
              <div className="p-8">
                 <div className="space-y-10">
                    <div>
                       <div className="flex justify-between items-center mb-3">
                          <span className="label-secondary text-[10px] opacity-60 uppercase font-black tracking-wide">MARGIN_UTILIZATION</span>
                          <span className="data-mono text-[13px] text-mint font-black">34.22%</span>
                       </div>
                       <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                          <motion.div 
                            initial={{ width: 0 }}
                            animate={{ width: '34.22%' }}
                            transition={{ duration: 1, delay: 0.8 }}
                            className="h-full bg-mint shadow-[0_0_12px_rgba(0,255,163,0.5)]" 
                          />
                       </div>
                    </div>
                    
                    <div>
                       <div className="flex justify-between items-center mb-3">
                          <span className="label-secondary text-[10px] opacity-60 uppercase font-black tracking-wide">LIQUIDITY_RESERVE_FACTOR</span>
                          <span className="data-mono text-[13px] text-teal font-black">8.45x</span>
                       </div>
                       <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                          <motion.div 
                            initial={{ width: 0 }}
                            animate={{ width: '85%' }}
                            transition={{ duration: 1, delay: 1 }}
                            className="h-full bg-teal shadow-[0_0_12px_rgba(20,184,166,0.5)]" 
                          />
                       </div>
                    </div>

                    <div className="mt-8 p-6 bg-purple-500/5 border border-purple-500/10 relative overflow-hidden rounded-xl">
                       <div className="absolute top-0 left-0 h-full w-1.5 bg-purple-500" />
                       <p className="text-[12px] leading-relaxed text-white/70">
                          <span className="text-purple-400 font-black tracking-[0.2em] uppercase mr-3">THREAT_LEVEL: NOMINAL.</span>
                          Concentration in high-beta assets has shifted +4.2% since previous epoch. Re-calibration recommended if volatility exceeds 2.5σ.
                       </p>
                    </div>
                 </div>
              </div>
           </AnimatedCard>
        </div>
      </motion.div>
    </div>
  );
};

export default RiskManagementPage;
