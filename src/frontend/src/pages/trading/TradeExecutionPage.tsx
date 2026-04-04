import React from 'react';
import { DepthOfMarket } from '../../features/trading/components/DepthOfMarket';
import { LevelIIQuotes } from '../../features/trading/components/LevelIIQuotes';
import { OrderTicket } from '../../features/trading/components/OrderTicket';
import { TimeAndSales } from '../../features/trading/components/TimeAndSales';
import { AnimatedCard } from '../../components/common/AnimatedCard';

const TradeExecutionPage: React.FC = () => {
  return (
    <div className="p-6 h-[calc(100vh-64px)] flex flex-col gap-4 bg-bento-bg overflow-hidden">
      <div className="flex-1 flex gap-4 min-h-0">
        {/* Depth of Market - Tactical Column */}
        <div className="flex-[1.2] flex flex-col min-h-0">
          <AnimatedCard className="h-full !p-0 border-mint/20 hover:border-mint/40 transition-colors overflow-hidden">
            <div className="p-4 border-b border-white/5 bg-white/2 flex justify-between items-center">
              <span className="label-secondary opacity-60">DEPTH_OF_MARKET // ORDER_FLOW</span>
              <div className="status-pill text-[10px] bg-mint/10 text-mint border-mint/20">LIVE_L2</div>
            </div>
            <div className="flex-1 overflow-auto p-2">
              <DepthOfMarket />
            </div>
          </AnimatedCard>
        </div>

        {/* Level II & Order Entry - Execution Core */}
        <div className="flex-1 flex flex-col gap-4 min-h-0">
           <AnimatedCard className="flex-1 !p-0 border-purple/20 hover:border-purple/40 transition-colors overflow-hidden">
             <div className="p-4 border-b border-white/5 bg-white/2">
                <span className="label-secondary opacity-60">LEVEL_II_QUOTES</span>
             </div>
             <div className="flex-1 overflow-auto p-2">
                <LevelIIQuotes />
             </div>
           </AnimatedCard>

           <AnimatedCard className="flex-[0.8] !p-0 border-amber/20 hover:border-amber/40 transition-colors overflow-hidden">
              <div className="p-4 border-b border-white/5 bg-white/2">
                 <span className="label-secondary opacity-60">DIRECT_EXECUTION_TICKET</span>
              </div>
              <div className="flex-1 overflow-auto p-4">
                 <OrderTicket />
              </div>
           </AnimatedCard>
        </div>
      </div>

      {/* Time & Sales Telemetry */}
      <div className="h-48">
        <AnimatedCard className="h-full !p-0 border-teal/20 hover:border-teal/40 transition-colors overflow-hidden">
           <div className="p-3 border-b border-white/5 bg-white/2 flex justify-between items-center">
              <span className="label-secondary opacity-60">TERMINAL_EXECUTION_HISTORY</span>
              <span className="data-mono text-[10px] text-white/40 uppercase tracking-widest">Connected_Via_Manifold_v2</span>
           </div>
           <div className="h-full overflow-hidden">
              <TimeAndSales />
           </div>
        </AnimatedCard>
      </div>
    </div>
  );
};

export default TradeExecutionPage;
