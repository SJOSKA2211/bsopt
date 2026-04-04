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
    <div className="p-6 h-[calc(100vh-64px)] overflow-auto bg-bento-bg">
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="bento-grid"
      >
        {/* Key Metrics Row */}
        <div className="col-span-12">
           <AnimatedCard className="p-8 overflow-hidden relative border-white/5 hover:border-white/10 transition-colors">
              <div className="absolute inset-0 bg-bento-grid bg-[length:32px_32px] opacity-[0.03] pointer-events-none" />
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-8 relative z-10">
                 {[
                   { label: 'NET_LIQUIDATION_VALUE', value: '$254,120.42', color: '#00FFA3', trend: '+0.85%' },
                   { label: 'DAILY_UNREALIZED_P&L', value: '+$3,420.12', color: '#00FFA3', trend: '+1.4%' },
                   { label: 'MAINTENANCE_MARGIN_CAP', value: '$42,500.00', color: '#8B5CF6', trend: 'NOMINAL' },
                   { label: 'EXCESS_LIQUIDITY_BUFFER', value: '$211,620.42', color: '#14B8A6', trend: 'SECURE' },
                 ].map((m, idx) => (
                   <div key={m.label} className="relative group">
                      <span className="label-secondary text-[9px] opacity-40 mb-2 block tracking-widest">{m.label}</span>
                      <div className="data-mono text-2xl font-black text-white tracking-tighter mb-2">{m.value}</div>
                      <div className="flex items-center gap-2">
                         <div className="status-pill text-[8px] bg-white/5 border-white/10 text-white/60 group-hover:border-white/20 transition-colors" style={{ borderLeftColor: m.color, borderLeftWidth: 2 }}>
                            {m.trend}
                         </div>
                      </div>
                      {idx < 3 && <div className="hidden md:block absolute right-[-16px] top-[20%] h-[60%] w-[1px] bg-white/5" />}
                   </div>
                 ))}
              </div>
           </AnimatedCard>
        </div>

        {/* Performance Chart */}
        <div className="col-span-12 lg:col-span-8">
           <AnimatedCard className="h-[480px] !p-0 border-purple/20 hover:border-purple/40 transition-colors overflow-hidden relative">
              <div className="p-4 border-b border-white/5 bg-white/2 flex justify-between items-center">
                 <span className="label-secondary opacity-40">EQUITY_CURVE_TELEMETRY // HISTORICAL_DATA_v8</span>
                 <div className="flex gap-1">
                    {['1D', '1W', '1M', '3M', 'YTD', 'ALL'].map(t => (
                       <button key={t} className={`px-2.5 py-1 text-[9px] font-black border transition-colors ${t === '3M' ? 'bg-mint/10 border-mint/20 text-mint' : 'bg-black/30 border-white/5 text-white/40 hover:bg-white/5'}`}>
                          {t}
                       </button>
                    ))}
                 </div>
              </div>
              <div className="p-2 h-[calc(100%-56px)]">
                 <EquityCurveChart />
              </div>
           </AnimatedCard>
        </div>

        {/* Portfolio Greeks */}
        <div className="col-span-12 lg:col-span-4">
           <AnimatedCard className="h-[480px] !p-0 border-white/5 hover:border-white/10 transition-colors overflow-hidden relative">
              <div className="p-4 border-b border-white/5 bg-white/2">
                 <span className="label-secondary opacity-40">AGGREGATED_PORTFOLIO_GREEKS</span>
              </div>
              <div className="p-6">
                 <div className="space-y-8">
                   {[
                     { label: 'NET_DELTA_EXPOSURE', value: '+42.52 Δ', percent: 65, color: '#00FFA3' },
                     { label: 'GAMMA_ACCELERATION', value: '+1.240 Γ', percent: 45, color: '#8B5CF6' },
                     { label: 'THETA_DECAY_ABS', value: '-245.12 Θ', percent: 72, color: '#EF4444' },
                     { label: 'VEGA_VOL_SENSITIVITY', value: '+142.05 V', percent: 30, color: '#14B8A6' },
                   ].map(g => (
                     <div key={g.label}>
                        <div className="flex justify-between items-center mb-2">
                           <span className="label-secondary text-[9px] opacity-60">{g.label}</span>
                           <span className="data-mono text-[12px] font-black" style={{ color: g.color }}>{g.value}</span>
                        </div>
                        <div className="h-0.5 w-full bg-white/5 rounded-full overflow-hidden">
                           <motion.div 
                             initial={{ width: 0 }}
                             animate={{ width: `${g.percent}%` }}
                             transition={{ duration: 1, delay: 0.5 }}
                             className="h-full"
                             style={{ backgroundColor: g.color, boxShadow: `0 0 10px ${g.color}66` }} 
                           />
                        </div>
                     </div>
                   ))}
                 </div>
                 
                 <div className="mt-10 p-4 bg-white/2 border border-white/5 relative overflow-hidden group">
                    <span className="label-secondary text-[8px] opacity-40 mb-2 block uppercase">GREEK_SYMMETRY_ANALysis</span>
                    <p className="text-[10px] leading-relaxed text-white/50 group-hover:text-white/70 transition-colors">
                       Portfolio is currently <span className="text-mint font-black tracking-widest uppercase">DELTA_NEUTRAL_BIAS</span>. Gamma exposure is concentrated in near-term expirations. Suggest re-balancing Vega exposure if IV exceeds 22%.
                    </p>
                    <div className="absolute bottom-[-10px] right-[-10px] w-12 h-12 bg-mint/5 blur-[20px] pointer-events-none" />
                 </div>
              </div>
           </AnimatedCard>
        </div>

        {/* Positions Section */}
        <div className="col-span-12">
           <AnimatedCard className="!p-0 border-white/5 hover:border-white/10 transition-colors overflow-hidden relative">
              <div className="absolute top-4 right-8 bg-amber text-black text-[9px] font-black px-3 py-1 skew-x-[-20deg] z-10">
                 REAL_TIME_INVENTORY
              </div>
              <div className="p-4 border-b border-white/5 bg-white/2">
                 <span className="label-secondary opacity-40 uppercase">ACTIVE_POSITIONS_CORE</span>
              </div>
              <div className="p-0">
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
