import React from 'react';
import { motion } from 'framer-motion';

const KpiCard = ({ label, value, color, prefix = '', index = 0 }: any) => (
  <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.05 }} className="bento-card relative overflow-hidden group">
     <div className="absolute top-0 right-0 w-16 h-16 bg-gradient-to-bl from-white/5 to-transparent pointer-events-none" />
     <span className="label-secondary opacity-60">{label}</span>
     <div className="flex items-baseline gap-1 mt-2">
        <span className="data-mono text-3xl font-black text-white group-hover:text-mint transition-colors">{prefix}{value}</span>
     </div>
     <div className="h-0.5 w-12 mt-4 rounded-full shadow-[0_0_10px_currentcolor]" style={{ backgroundColor: color, color }} />
  </motion.div>
);

const DashboardPage = () => (
  <div className="p-8 space-y-8 min-h-full">
    <div className="bento-grid">
       <div className="col-span-12 sm:col-span-6 lg:col-span-3">
          <KpiCard label="SYSTEM_GAMMA" value="2.412" color="#00FFA3" index={0} />
       </div>
       <div className="col-span-12 sm:col-span-6 lg:col-span-3">
          <KpiCard label="PORTFOLIO_NAV" value="254,120.42" prefix="$" color="#F59E0B" index={1} />
       </div>
       <div className="col-span-12 sm:col-span-6 lg:col-span-3">
          <KpiCard label="VEGA_SENS" value="4.12k" color="#14B8A6" index={2} />
       </div>
       <div className="col-span-12 sm:col-span-6 lg:col-span-3">
          <KpiCard label="MODEL_CONFIDENCE" value="98.4%" color="#BD00FF" index={3} />
       </div>

       <div className="col-span-12 lg:col-span-8">
          <div className="bento-card h-[520px] flex flex-col relative overflow-hidden group">
             <div className="p-4 border-b border-white/5 mb-8 flex justify-between items-center bg-white/[0.02]">
                <span className="label-secondary opacity-40">DEEP_INFERENCE_ENGINE // RT_PROBABILITY_DENSITY</span>
                <span className="status-pill text-[9px] healthy scale-90">COMPUTING</span>
             </div>
             <div className="flex-grow flex flex-col items-center justify-center opacity-10">
                <div className="w-32 h-32 border-2 border-mint rounded-full animate-pulse flex items-center justify-center">
                   <div className="w-16 h-16 border border-mint/40 rounded-full animate-ping" />
                </div>
                <span className="mt-8 text-[11px] font-black tracking-[1.5em] uppercase text-mint">SYSTEM_ACTIVE</span>
             </div>
          </div>
       </div>

       <div className="col-span-12 lg:col-span-4">
          <div className="bento-card h-[520px] !p-0 overflow-hidden border-white/5">
             <div className="p-6 border-b border-white/5 bg-white/[0.02]">
                <span className="label-secondary opacity-40">RISK_CONCENTRATION_MAP</span>
             </div>
             <div className="p-8 space-y-8">
                {[
                  { l: "EQUITY_VOL", v: 85, c: "#00FFA3" },
                  { l: "CREDIT_SPREAD", v: 42, c: "#F59E0B" },
                  { l: "FX_VARIANCE", v: 28, c: "#BD00FF" }
                ].map((r, i) => (
                  <div key={i} className="space-y-3">
                     <div className="flex justify-between text-[9px] font-black tracking-widest text-white/40 uppercase">
                        <span>{r.l}</span>
                        <span style={{ color: r.c }}>{r.v}%</span>
                     </div>
                     <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                        <motion.div initial={{ width: 0 }} animate={{ width: `${r.v}%` }} transition={{ duration: 1, delay: i * 0.1 }} className="h-full shadow-[0_0_10px_currentcolor]" style={{ backgroundColor: r.c, color: r.c }} />
                     </div>
                  </div>
                ))}
             </div>
          </div>
       </div>
    </div>
  </div>
);
export default DashboardPage;
