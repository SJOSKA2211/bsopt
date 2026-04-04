import React, { useMemo } from 'react';
import { usePricingStore, type PricingState } from '../../../store/usePricingStore';

export const RiskExposureGrid: React.FC = () => {
  const prices = usePricingStore((state: PricingState) => state.prices);
  
  // Simulated risk matrix based on live prices for high-fidelity UI feedback
  const data = useMemo(() => {
    const symbols = Object.keys(prices);
    if (symbols.length === 0) return Array(16).fill(0.1);
    
    // Derived values for the 4x4 matrix
    return Array.from({ length: 16 }, (_, i) => {
      const priceVal = prices[symbols[i % symbols.length]]?.price || 100;
      return (Math.sin(i + priceVal) * 0.8);
    });
  }, [prices]);

  const getColorClass = (val: number) => {
    if (val > 0.4) return 'bg-mint/40 text-black border-mint/20';
    if (val > 0) return 'bg-mint/10 text-mint border-mint/10';
    if (val < -0.4) return 'bg-red-500/40 text-black border-red-500/20';
    return 'bg-red-500/10 text-red-400 border-red-500/10';
  };

  return (
    <div className="h-full flex flex-col">
      <div className="grid grid-cols-4 gap-1 auto-rows-fr flex-grow min-h-0">
        {data.map((val, i) => (
          <div 
            key={i} 
            className={`flex items-center justify-center rounded-md border transition-all duration-500 ${getColorClass(val)}`}
          >
            <span className="data-mono text-[10px] font-black">
              {(val * 10).toFixed(1)}k
            </span>
          </div>
        ))}
      </div>
      <div className="mt-4 flex justify-between items-center opacity-40">
        <span className="text-[9px] font-bold uppercase tracking-tighter">Delta_Exposure</span>
        <span className="text-[9px] font-bold uppercase tracking-tighter">Gamma_Sens</span>
      </div>
    </div>
  );
};
