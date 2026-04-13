import { usePricingStore, type PricingState } from '../../../store/usePricingStore';
import { motion } from 'framer-motion';
import { useMLInference, useComparisonData } from '../../../api/hooks';
import { CircularProgress, Box } from '@mui/material';

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

interface DeepInferenceEngineProps {
  symbol: string;
}

export const DeepInferenceEngine: React.FC<DeepInferenceEngineProps> = ({ symbol }) => {
  const mlAccuracy = usePricingStore((state: PricingState) => state.mlAccuracy);
  const { data: inferenceData, loading: isInferenceLoading } = useMLInference(symbol);
  const { data: comparisonData, isLoading: isComparisonLoading } = useComparisonData();
  
  if (isInferenceLoading || isComparisonLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', h: '100%' }}>
        <CircularProgress size={20} sx={{ color: '#00FFA3' }} />
      </Box>
    );
  }

  const prediction = inferenceData?.mlPrediction;
  const confidence = prediction?.confidence_interval ? (prediction.confidence_interval * 100) : 85;
  const latestAccuracy = comparisonData?.data?.accuracy || mlAccuracy || 0;

  return (
    <div className="flex flex-col h-full">
      <div className="space-y-4">
        <ModelMetric 
          label="Directional Prob (Bullish)" 
          value={`${latestAccuracy.toFixed(1)}%`} 
          percent={latestAccuracy} 
          color="#00FFA3" 
        />
        <ModelMetric 
          label="Model Confidence" 
          value={`${confidence.toFixed(1)}%`} 
          percent={confidence} 
          color="#8B5CF6" 
        />
        <ModelMetric 
          label="Signal Strength" 
          value={confidence > 80 ? "High" : "Moderate"} 
          percent={confidence} 
          color="#14B8A6" 
        />
        
        <div className="mt-6 p-4 bg-mint/5 border-l-2 border-mint rounded-r-lg">
          <span className="label-secondary text-[9px] block mb-2 opacity-60">LATEST_ML_INSIGHT</span>
          <p className="text-[11px] font-medium leading-relaxed text-white/90">
            {prediction?.model_name || 'XGB-Hybrid'} suggests {prediction?.predicted_price ? `target price of $${prediction.predicted_price.toFixed(2)}` : 'potential mean reversion'}. 
            Accuracy baseline: <span className="text-mint font-bold">{latestAccuracy.toFixed(1)}%</span>.
          </p>
        </div>
      </div>
    </div>
  );
};
