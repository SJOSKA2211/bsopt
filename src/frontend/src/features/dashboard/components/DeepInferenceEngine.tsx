import React from 'react';
import { usePricingStore, type PricingState } from '../../../store/usePricingStore';
import { motion } from 'framer-motion';

interface MetricProps {
  label: string;
  value: string;
  percent: number;
  color: string;
}

const ModelMetric: React.FC<MetricProps> = ({ label, value, percent, color }) => (
  <div className="mb-4">
    <div className="flex justify-between mb-1">
      <span className="label-secondary text-[10px] opacity-50 uppercase tracking-widest">{label}</span>
      <span className="data-mono text-[11px] font-black" style={{ color }}>{value}</span>
    </div>
    <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden border border-white/5">
      <motion.div 
        initial={{ width: 0 }}
        animate={{ width: `${percent}%` }}
        transition={{ duration: 1, ease: "easeOut" }}
        className="h-full rounded-full"
        style={{ backgroundColor: color }}
      />
    </div>
  </div>
);

export const DeepInferenceEngine: React.FC = () => {
  const mlAccuracy = usePricingStore((state: PricingState) => state.mlAccuracy);
  
  return (
    <div className="flex flex-col h-full">
      <div className="space-y-4">
        <ModelMetric 
          label="Directional Prob (Bullish)" 
          value={`${mlAccuracy.toFixed(1)}%`} 
          percent={mlAccuracy} 
          color="#00FFA3" 
        />
        <ModelMetric 
          label="Vol Expansion Confidence" 
          value="82.1%" 
          percent={82.1} 
          color="#8B5CF6" 
        />
        <ModelMetric 
          label="Signal Strength" 
          value="High" 
          percent={90} 
          color="#14B8A6" 
        />
        
        <div className="mt-6 p-4 bg-mint/5 border-l-2 border-mint rounded-r-lg">
          <span className="label-secondary text-[9px] block mb-2 opacity-60">LATEST_ML_INSIGHT</span>
          <p className="text-[11px] font-medium leading-relaxed text-white/90">
            Gamma-weighted distribution suggests potential 1.2% mean reversion within 4 hours. Accuracy baseline: <span className="text-mint font-bold">{mlAccuracy.toFixed(1)}%</span>.
          </p>
        </div>
      </div>
    </div>
  );
};
