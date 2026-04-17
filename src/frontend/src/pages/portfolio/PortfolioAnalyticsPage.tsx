import React from 'react';
import { EquityCurveChart } from '../../features/portfolio/components/EquityCurveChart';
import { ActivePositionsTable } from '../../features/portfolio/components/ActivePositionsTable';
import { RecentTradeActivity } from '../../features/portfolio/components/RecentTradeActivity';
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

const PortfolioAnalyticsPage: React.FC = () => {
  return (
    <div className="p-6 h-full overflow-auto bg-bento-bg">
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="bento-grid"
      >
        {/* Key Metrics Row */}
        <div className="col-span-12">
           <AnimatedCard className="p-10 overflow-hidden relative border-white/5 hover:border-white/10 transition-all bg-white/[0.02]">
              <div className="absolute inset-0 bg-bento-grid bg-[length:32px_32px] opacity-[0.03] pointer-events-none" />
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-12 relative z-10">
                 {[
                   { label: 'NET_LIQUIDATION_VALUE', value: '$254,120.42', color: '#00FFA3', trend: '+0.85%' },
                   { label: 'DAILY_UNREALIZED_P&L', value: '+$3,420.12', color: '#00FFA3', trend: '+1.4%' },
                   { label: 'MAINTENANCE_MARGIN_CAP', value: '$42,500.00', color: '#8B5CF6', trend: 'NOMINAL' },
                   { label: 'EXCESS_LIQUIDITY_BUFFER', value: '$211,620.42', color: '#14B8A6', trend: 'SECURE' },
                 ].map((m, idx) => (
                   <div key={m.label} className="relative group">
                      <span className="label-secondary text-[9px] opacity-40 mb-3 block tracking-widest leading-none">{m.label}</span>
                      <div className="data-mono text-3xl font-black text-white tracking-tighter mb-3">{m.value}</div>
                      <div className="flex items-center gap-2">
                         <div className="status-pill text-[9px] bg-white/5 border-white/10 text-white/70 group-hover:border-white/20 transition-all" style={{ borderLeftColor: m.color, borderLeftWidth: 3 }}>
                            {m.trend}
                         </div>
                      </div>
                      {idx < 3 && <div className="hidden lg:block absolute right-[-24px] top-[15%] h-[70%] w-[1px] bg-white/5" />}
                   </div>
                 ))}
              </div>
           </AnimatedCard>
        </div>

        {/* Performance Chart */}
        <div className="col-span-12 lg:col-span-8">
           <AnimatedCard className="h-[520px] !p-0 border-purple/20 hover:border-purple/40 transition-colors overflow-hidden relative bg-white/[0.01]">
              <div className="p-6 border-b border-white/5 bg-white/2 flex justify-between items-center">
                 <span className="label-secondary opacity-40">EQUITY_CURVE_TELEMETRY // HISTORICAL_DATA</span>
                 <div className="flex gap-2">
                    {['1D', '1W', '1M', '3M', 'YTD', 'ALL'].map(t => (
                       <button key={t} className={`px-4 py-1.5 text-[10px] rounded-lg font-black border transition-all ${t === '3M' ? 'bg-mint/10 border-mint/20 text-mint' : 'bg-black/30 border-white/5 text-white/40 hover:bg-white/5'}`}>
                          {t}
                       </button>
                    ))}
                 </div>
              </div>
              <div className="p-4 h-[calc(100%-72px)]">
                 <EquityCurveChart />
              </div>
           </AnimatedCard>
        </div>

        {/* Portfolio Greeks */}
        <div className="col-span-12 lg:col-span-4">
           <AnimatedCard className="h-[520px] !p-0 border-white/5 hover:border-white/10 transition-colors overflow-hidden relative">
              <div className="p-6 border-b border-white/5 bg-white/2">
                 <span className="label-secondary opacity-40">AGGREGATED_PORTFOLIO_GREEKS</span>
              </div>
              <div className="p-8">
                 <div className="space-y-10">
                   {[
                     { label: 'NET_DELTA_EXPOSURE', value: '+42.52 Δ', percent: 65, color: '#00FFA3' },
                     { label: 'GAMMA_ACCELERATION', value: '+1.240 Γ', percent: 45, color: '#8B5CF6' },
                     { label: 'THETA_DECAY_ABS', value: '-245.12 Θ', percent: 72, color: '#EF4444' },
                     { label: 'VEGA_VOL_SENSITIVITY', value: '+142.05 V', percent: 30, color: '#14B8A6' },
                   ].map(g => (
                     <div key={g.label}>
                        <div className="flex justify-between items-center mb-3">
                           <span className="label-secondary text-[10px] opacity-60 tracking-wider font-extrabold">{g.label}</span>
                           <span className="data-mono text-[14px] font-black" style={{ color: g.color }}>{g.value}</span>
                        </div>
                        <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                           <motion.div 
                             initial={{ width: 0 }}
                             animate={{ width: `${g.percent}%` }}
                             transition={{ duration: 1, delay: 0.5 }}
                             className="h-full"
                             style={{ backgroundColor: g.color, boxShadow: `0 0 12px ${g.color}44` }} 
                           />
                        </div>
                     </div>
                   ))}
                 </div>
                 
                 <div className="mt-12 p-5 bg-white/2 border border-white/5 relative overflow-hidden group rounded-xl">
                    <span className="label-secondary text-[9px] opacity-40 mb-3 block uppercase font-black">GREEK_SYMMETRY_SUMMARY</span>
                    <p className="text-[11px] leading-relaxed text-white/50 group-hover:text-white/80 transition-colors">
                       Portfolio is currently <span className="text-mint font-black tracking-widest uppercase">DELTA_NEUTRAL_BIAS</span>. Gamma exposure is concentrated in near-term expirations.
                    </p>
                    <div className="absolute bottom-[-10px] right-[-10px] w-14 h-14 bg-mint/5 blur-[24px] pointer-events-none" />
                 </div>
              </div>
           </AnimatedCard>
        </div>

        {/* Positions Section */}
        <div className="col-span-12">
           <AnimatedCard className="!p-0 border-white/5 hover:border-white/10 transition-colors overflow-hidden relative bg-white/[0.01]">
              <div className="absolute top-0 right-12 bg-amber text-black text-[10px] font-black px-6 py-1.5 skew-x-[-15deg] z-10 shadow-[0_0_20px_rgba(255,170,0,0.3)]">
                 REAL_TIME_INVENTORY
              </div>
              <div className="p-6 border-b border-white/5 bg-white/2">
                 <span className="label-secondary opacity-40 uppercase tracking-[0.2em] font-black">ACTIVE_POSITIONS_CORE</span>
              </div>
              <div className="p-4">
                 <ActivePositionsTable />
              </div>
           </AnimatedCard>
        </div>

        {/* Bottom Activity Section */}
        <div className="col-span-12">
           <AnimatedCard className="!p-0 border-teal/20 hover:border-teal/40 transition-colors overflow-hidden relative">
              <div className="p-4 border-b border-white/5 bg-white/2">
                 <span className="label-secondary opacity-40 uppercase">RECENT_EXECUTION_SEQUENCE</span>
              </div>
              <div className="p-0">
                 <RecentTradeActivity />
              </div>
           </AnimatedCard>
        </div>
      </motion.div>
    </div>
  );
};

export default PortfolioAnalyticsPage;
