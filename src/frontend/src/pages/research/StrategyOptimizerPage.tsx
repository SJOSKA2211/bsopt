import React from 'react';
import { OptimizationControls } from '../../features/research/components/OptimizationControls';
import { OptimalConfigCard } from '../../features/research/components/OptimalConfigCard';
import { SweepResultsTable } from '../../features/research/components/SweepResultsTable';
import VolatilitySurface3D from '../../features/options/components/VolatilitySurface3D';
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

const StrategyOptimizerPage: React.FC = () => {
  return (
    <div className="p-6 h-[calc(100vh-64px)] overflow-auto bg-bento-bg">
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="bento-grid"
      >
        {/* Header Shard */}
        <div className="col-span-12">
           <AnimatedCard className="p-6 overflow-hidden relative border-mint/20 hover:border-mint/40 transition-colors">
              <div className="absolute top-[-40px] right-[-40px] w-64 h-64 bg-mint/5 rounded-full blur-[100px] pointer-events-none" />
              
              <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-6 relative z-10">
                 <div>
                    <span className="label-secondary text-mint mb-2 block tracking-[0.2em] font-black">
                       STRATEGY_GENESIS // GEOMETRIC_OPTIMIZER_v4.2
                    </span>
                    <div className="flex flex-wrap items-baseline gap-4">
                       <h1 className="data-mono text-4xl font-black text-white tracking-tighter">
                          AAPL // BULL_CALL_SPREAD
                       </h1>
                       <div className="status-pill bg-amber/10 text-amber border-amber/20 scale-90">RUN_SCAN_SEQUENCE_042</div>
                    </div>
                 </div>
                 <button className="h-12 px-8 bg-mint text-black font-black text-[11px] uppercase tracking-widest rounded-none shadow-[0_0_25px_rgba(0,255,163,0.3)] hover:bg-mint/90 transition-all active:scale-95">
                    EXECUTE_HEURISTIC_OPTIMIZATION
                 </button>
              </div>
           </AnimatedCard>
        </div>

        {/* Controls Bar */}
        <div className="col-span-12">
           <AnimatedCard className="!p-0 border-white/5 hover:border-white/10 transition-colors overflow-hidden">
              <div className="p-4 border-b border-white/5 bg-white/2">
                 <span className="label-secondary opacity-40">INPUT_PARAMETERS // CONSTRAINTS</span>
              </div>
              <div className="p-6">
                 <OptimizationControls />
              </div>
           </AnimatedCard>
        </div>

        {/* 3D Manifold */}
        <div className="col-span-12 lg:col-span-8">
           <AnimatedCard className="h-[500px] !p-0 border-purple/20 hover:border-purple/40 transition-colors overflow-hidden relative">
              <div className="p-4 border-b border-white/5 bg-white/2">
                 <span className="label-secondary opacity-40">OPTIMIZATION_MANIFOLD // 3D_HYPER_SURFACE_SCAN</span>
              </div>
              <div className="p-2 h-[calc(100%-56px)] relative">
                 <VolatilitySurface3D symbol="AAPL" />
                 <div className="absolute bottom-4 left-4 p-2 bg-black/60 border border-white/10 backdrop-blur-md rounded">
                    <span className="data-mono text-[9px] text-teal font-black uppercase tracking-widest">COORD_SYSTEM: CARTESIAN_v3 // SCALE: 1.2x</span>
                 </div>
              </div>
           </AnimatedCard>
        </div>

        {/* Optimal Configs */}
        <div className="col-span-12 lg:col-span-4">
           <AnimatedCard className="h-[500px] !p-0 border-white/5 hover:border-white/10 transition-colors overflow-hidden relative">
              <div className="p-4 border-b border-white/5 bg-white/2">
                 <span className="label-secondary opacity-40">TOP_RANKED_CONFIGURATIONS</span>
              </div>
              <div className="p-6 h-[calc(100%-56px)] overflow-auto space-y-4">
                 {[
                   { id: 1, strike1: 190.0, strike2: 195.0, change: '+24.52%', score: '2.42' },
                   { id: 2, strike1: 187.5, strike2: 192.5, change: '+21.24%', score: '2.18' },
                   { id: 3, strike1: 192.5, strike2: 197.5, change: '+19.95%', score: '1.95' },
                   { id: 4, strike1: 190.0, strike2: 200.0, change: '+28.12%', score: '1.82' },
                 ].map(config => (
                    <OptimalConfigCard key={config.id} {...config} />
                 ))}
              </div>
           </AnimatedCard>
        </div>

        {/* Bottom Sweep Results */}
        <div className="col-span-12">
           <AnimatedCard className="!p-0 border-white/5 hover:border-white/10 transition-colors overflow-hidden">
              <div className="p-4 border-b border-white/5 bg-white/2">
                 <span className="label-secondary opacity-40">PARAMETER_SWEEP_SEQUENCE // DETAILED_TELEMETRY</span>
              </div>
              <div className="p-0">
                 <SweepResultsTable />
              </div>
           </AnimatedCard>
        </div>
      </motion.div>
    </div>
  );
};

export default StrategyOptimizerPage;
