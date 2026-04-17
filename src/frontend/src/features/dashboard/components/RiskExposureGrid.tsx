import React, { useMemo } from 'react';
import { usePricingStore, type PricingState } from '../../../store/usePricingStore';
import { useHeatmap } from '../../../api/hooks';
import { CircularProgress } from '@mui/material';

export const RiskExposureGrid: React.FC = () => {
  const prices = usePricingStore((state: PricingState) => state.prices);
  
  // Use the first symbol or SPX as a baseline for the global system risk view
  const symbols = Object.keys(prices);
  const baselineSymbol = symbols.includes('SPX') ? 'SPX' : (symbols[0] || 'SPX');
  const baselinePrice = prices[baselineSymbol]?.price || 4400;

  const heatmapRequest = useMemo(() => ({
    spot: baselinePrice,
    strike: baselinePrice,
    time_to_expiry: 0.1, // 10% of year ~ 1 month
    volatility: 0.2,
    rate: 0.05,
    option_type: 'call',
    price_shifts: [-2, -1, 1, 2],
    vol_shifts: [-2, -1, 1, 2],
  }), [baselinePrice]);

  const { data: heatmap, isLoading } = useHeatmap(heatmapRequest);
  
  const flatData = useMemo(() => {
    if (!heatmap?.grid) return Array(16).fill(0);
    return heatmap.grid.flat().map((cell: any) => cell.pnl / 10); // Scale down for UI
  }, [heatmap]);

  const getColorClass = (val: number) => {
    if (val > 0.4) return 'bg-mint/40 text-black border-mint/20';
    if (val > 0) return 'bg-mint/10 text-mint border-mint/10';
    if (val < -0.4) return 'bg-red-500/40 text-black border-red-500/20';
    return 'bg-red-500/10 text-red-400 border-red-500/10';
  };

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <CircularProgress size={20} sx={{ color: '#00FFA3' }} />
      </div>
    );
  }

  return (
    <div data-testid="risk-exposure-grid-container" className="h-full flex flex-col">
      <div className="grid grid-cols-4 gap-1 auto-rows-fr flex-grow min-h-0">
        {flatData.map((val: number, i: number) => (
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
        <span className="text-[9px] font-bold uppercase tracking-tighter">Delta_Exposure (Shift %)</span>
        <span className="text-[9px] font-bold uppercase tracking-tighter">Heatmap_Sim</span>
      </div>
    </div>
  );
};
