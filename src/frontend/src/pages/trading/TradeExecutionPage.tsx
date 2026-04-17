import React from 'react';
import { motion } from 'framer-motion';
import { AnimatedCard } from '../../components/common/AnimatedCard';

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1 }
  }
};

export const TradeExecutionPage: React.FC = () => {
  return (
    <div className="p-6 bg-bento-bg">
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="bento-grid"
      >
        {/* Market Depth & Order Flow */}
        <div className="col-span-12 lg:col-span-8">
           <AnimatedCard className="h-[600px] !p-0 border-white/5 bg-white/[0.01] overflow-hidden relative">
              <div className="p-4 border-b border-white/5 bg-white/2 flex justify-between items-center">
                 <span className="label-secondary opacity-40">INSTITUTIONAL_ORDER_FLOW // DEPTH_OF_MARKET</span>
                 <div className="flex gap-4 items-center">
                    <div className="flex items-center gap-2">
                       <span className="text-[10px] text-white/30 uppercase font-black">VOL_PROFILE:</span>
                       <span className="text-[10px] text-mint font-black">HIGH</span>
                    </div>
                    <div className="status-pill healthy scale-90">LIVE_DATA</div>
                 </div>
              </div>
              <div className="p-6 h-[calc(100%-64px)] flex flex-col items-center justify-center relative overflow-hidden">
                 <div className="absolute inset-0 bg-bento-grid bg-[length:24px_24px] opacity-[0.02]" />
                 <div className="relative z-10 text-center space-y-4">
                    <span className="label-secondary opacity-20 text-[11px] font-black tracking-[0.4em]">
                       STREAMING_ORDER_INTERCEPTOR_INITIALIZING...
                    </span>
                    <div className="flex justify-center gap-1">
                       {[...Array(5)].map((_, i) => (
                          <motion.div
                             key={i}
                             animate={{ height: [4, 16, 8, 20, 4] }}
                             transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.1 }}
                             className="w-[2px] bg-mint/40"
                          />
                       ))}
                    </div>
                 </div>
              </div>
           </AnimatedCard>
        </div>

        {/* Execution Ticket */}
        <div className="col-span-12 lg:col-span-4">
           <div className="flex flex-col gap-6 h-[600px]">
              <AnimatedCard className="flex-grow !p-0 border-mint/20 bg-mint/[0.01] overflow-hidden relative group">
                 <div className="absolute top-0 left-0 w-full h-[2px] bg-mint shadow-[0_0_10px_#00ffa3] opacity-40" />
                 <div className="p-5 border-b border-white/5 bg-white/2">
                    <span className="label-secondary opacity-40 text-xs font-black">DIRECT_EXECUTION_TICKET</span>
                 </div>
                 <div className="p-8 space-y-8">
                    <div className="space-y-3">
                       <span className="label-secondary text-[10px] opacity-40 font-black tracking-widest leading-none">SELECTION</span>
                       <div className="text-2xl font-black text-white tracking-widest">SPY_DEC24_510_P</div>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4">
                       <button className="py-4 rounded-xl bg-mint/10 border border-mint/20 text-mint font-black tracking-widest hover:bg-mint/20 transition-all">BUY_LIMIT</button>
                       <button className="py-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-500 font-black tracking-widest hover:bg-red-500/20 transition-all">SELL_LIMIT</button>
                    </div>

                    <div className="p-5 rounded-xl bg-white/2 border border-white/5 space-y-4">
                       <div className="flex justify-between items-center px-1">
                          <span className="text-[10px] font-black text-white/30 uppercase">QUANTITY</span>
                          <span className="data-mono text-sm font-black text-white">42_CONTRACTS</span>
                       </div>
                       <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                          <div className="h-full w-[45%] bg-mint shadow-[0_0_10px_#00ffa3]" />
                       </div>
                    </div>

                    <div className="pt-6 border-t border-white/5">
                       <button className="w-full py-5 rounded-xl bg-gradient-to-r from-mint/80 to-quantum/80 text-black font-black tracking-[0.2em] transform active:scale-[0.98] transition-all shadow-[0_4px_20px_rgba(0,255,163,0.2)]">
                          TRANSMIT_ORDER_SEQUENCE
                       </button>
                    </div>
                 </div>
              </AnimatedCard>

              <AnimatedCard className="h-[180px] !p-0 border-white/5 bg-white/[0.01] overflow-hidden">
                 <div className="p-4 border-b border-white/5 bg-white/2">
                    <span className="label-secondary opacity-40">EXECUTION_STATUS_TELEMETRY</span>
                 </div>
                 <div className="p-6 flex flex-col justify-center gap-3">
                    <div className="flex justify-between items-center text-[11px] font-black text-white/40 uppercase tracking-widest">
                       <span>LATENCY (NY4)</span>
                       <span className="text-mint">4.12ms</span>
                    </div>
                    <div className="flex justify-between items-center text-[11px] font-black text-white/40 uppercase tracking-widest">
                       <span>ORCHESTRATOR_ACK</span>
                       <span className="text-mint">VERIFIED</span>
                    </div>
                 </div>
              </AnimatedCard>
           </div>
        </div>

        {/* Recently Filled Section */}
        <div className="col-span-12">
            <AnimatedCard className="!p-0 border-white/5 hover:border-white/10 transition-all overflow-hidden relative">
               <div className="p-5 border-b border-white/5 bg-white/2">
                  <span className="label-secondary opacity-40 uppercase">LIVE_OMNIBUS_ACTIVITY</span>
               </div>
               <div className="p-12 text-center">
                  <span className="label-secondary opacity-20 text-[10px] uppercase font-black tracking-[0.5em]">
                     CONNECTING_TO_INSTITUTIONAL_MESSAGING_BUS...
                  </span>
               </div>
            </AnimatedCard>
        </div>
      </motion.div>
    </div>
  );
};

export default TradeExecutionPage;
