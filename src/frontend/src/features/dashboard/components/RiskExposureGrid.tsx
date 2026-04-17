import React from 'react';
import { motion } from 'framer-motion';

export const RiskExposureGrid = () => (
  <div className="p-8 space-y-6">
     {[...Array(4)].map((_, i) => (
        <div key={i} className="space-y-2">
           <div className="flex justify-between text-[10px] font-black text-white/40 uppercase">
              <span>SECTOR_ALPHA_{i}</span>
              <span className="text-mint">{(85 - i * 15)}%</span>
           </div>
           <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
              <motion.div initial={{ width: 0 }} animate={{ width: `${85 - i * 15}%` }} className="h-full bg-mint shadow-[0_0_10px_#00ffa3]" />
           </div>
        </div>
     ))}
  </div>
);
