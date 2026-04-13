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
    <div className="p-6 h-[calc(100vh-64px)] overflow-auto bg-bento-bg">
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
           <AnimatedCard className="h-[500px] !p-0 border-white/5 hover:border-white/10 transition-colors overflow-hidden relative group">
              <div className="absolute inset-0 bg-bento-grid bg-[length:32px_32px] opacity-[0.03] group-hover:opacity-[0.05] transition-opacity" />
              <div className="p-4 border-b border-white/5 bg-white/2">
                 <span className="label-secondary opacity-40">QUANT_RISK_MANIFOLD // P&L_SURFACE_v3.2</span>
              </div>
              <div className="p-4 h-[calc(100%-56px)] flex items-center justify-center relative">
                 <div className="w-full h-full border border-white/5 bg-black/20 relative overflow-hidden flex items-center justify-center">
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
           <AnimatedCard className="!p-0 border-white/5 hover:border-white/10 transition-colors overflow-hidden relative">
              <div className="absolute top-0 left-6 px-3 py-1 bg-amber text-black text-[9px] font-black tracking-widest skew-x-[-15deg] z-10">
                 HEDGE_SUB_ORCHESTRATOR
              </div>
              <div className="p-4 pt-8 border-b border-white/5 bg-white/2">
                 <span className="label-secondary opacity-40">ACTIVE_DERIVATIVE_PROTECTION</span>
              </div>
              <div className="p-0">
                 {[
                   { symbol: 'SPY_241220_P_450', delta: '-124.5', type: 'PUT', status: 'ACTIVE' },
                   { symbol: 'VIX_250117_C_25', delta: '+45.2', type: 'CALL', status: 'WAITING' },
                   { symbol: 'TLT_250321_L_95', delta: '+12.0', type: 'EQUITY', status: 'ACTIVE' },
                 ].map((hedge, i) => (
                   <div key={i} className="flex items-center p-4 border-b border-white/5 hover:bg-white/2 transition-colors cursor-default">
                      <div className="flex-1">
                         <div className="text-[10px] font-black text-white tracking-widest">{hedge.symbol}</div>
                         <div className="label-secondary text-[8px] opacity-40 mt-1">{hedge.type} // STATUS: {hedge.status}</div>
                      </div>
                      <div className="flex-1 text-right">
                         <div className={`data-mono text-[11px] font-black ${hedge.delta.includes('-') ? 'text-red-500' : 'text-mint'}`}>
                            {hedge.delta} Δ
                         </div>
                         <div className="h-0.5 w-10 ml-auto mt-2 bg-white/10 rounded-full overflow-hidden">
                            <div className={`h-full w-[70%] ${hedge.delta.includes('-') ? 'bg-red-500' : 'bg-mint'}`} />
                         </div>
                      </div>
                   </div>
                 ))}
              </div>
           </AnimatedCard>
        </div>

        <div className="col-span-12 md:col-span-6">
           <AnimatedCard className="!p-0 border-white/5 hover:border-white/10 transition-colors overflow-hidden relative">
              <div className="p-4 border-b border-white/5 bg-white/2">
                 <span className="label-secondary opacity-40">MARGIN_LATENCY_MONITOR // CROSS_COLLATERAL</span>
              </div>
              <div className="p-6">
                 <div className="space-y-6">
                    <div>
                       <div className="flex justify-between items-center mb-2">
                          <span className="label-secondary text-[9px] opacity-60">MARGIN_UTILIZATION_RATIO</span>
                          <span className="data-mono text-[11px] text-mint font-black">34.22%</span>
                       </div>
                       <div className="h-0.5 w-full bg-white/5 rounded-full overflow-hidden">
                          <motion.div 
                            initial={{ width: 0 }}
                            animate={{ width: '34.22%' }}
                            transition={{ duration: 1, delay: 0.8 }}
                            className="h-full bg-mint shadow-[0_0_8px_rgba(0,255,163,0.4)]" 
                          />
                       </div>
                    </div>
                    
                    <div>
                       <div className="flex justify-between items-center mb-2">
                          <span className="label-secondary text-[9px] opacity-60">LIQUIDITY_RESERVE_FACTOR</span>
                          <span className="data-mono text-[11px] text-teal font-black">8.45x</span>
                       </div>
                       <div className="h-0.5 w-full bg-white/5 rounded-full overflow-hidden">
                          <motion.div 
                            initial={{ width: 0 }}
                            animate={{ width: '85%' }}
                            transition={{ duration: 1, delay: 1 }}
                            className="h-full bg-teal shadow-[0_0_8px_rgba(20,184,166,0.4)]" 
                          />
                       </div>
                    </div>

                    <div className="mt-4 p-4 bg-purple/3 border border-purple/10 relative overflow-hidden">
                       <div className="absolute top-0 left-0 h-full w-1 bg-purple" />
                       <p className="text-[11px] leading-relaxed text-white/70">
                          <span className="text-purple font-black tracking-widest uppercase mr-2.5">THREAT_LEVEL: NOMINAL.</span>
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
