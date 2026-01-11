import React, { useState, useEffect, useMemo } from 'react';
import { useWASMPricer } from '../hooks/useWASMPricer';
import { OptionParams, Greeks } from '../services/WASMPricingService';

interface OptionData {
  strike: number;
  price: number;
  greeks: Greeks;
}

export const OptionsChain: React.FC = () => {
  const { isInitialized, error, calculateGreeks, priceCall } = useWASMPricer();
  const [options, setOptions] = useState<OptionData[]>([]);
  const [loading, setLoading] = useState(true);
  const [benchTime, setBenchTime] = useState<number>(0);

  const spot = 100;
  const timeToMaturity = 0.25;
  const volatility = 0.25;
  const rate = 0.05;
  const dividend = 0.02;

  const strikes = useMemo(() => Array.from({ length: 21 }, (_, i) => 90 + i), []);

  useEffect(() => {
    async function loadOptions() {
      if (!isInitialized) return;

      const startTime = performance.now();
      
      const optionParams: OptionParams[] = strikes.map(strike => ({
        spot,
        strike,
        time: timeToMaturity,
        vol: volatility,
        rate,
        div: dividend,
      }));

      const results = await Promise.all(
        optionParams.map(async (params) => {
          const price = await priceCall(params);
          const greeks = await calculateGreeks(params);
          return { strike: params.strike, price, greeks };
        })
      );

      const endTime = performance.now();
      setBenchTime(endTime - startTime);
      setOptions(results);
      setLoading(false);
    }

    loadOptions();
  }, [isInitialized, strikes, calculateGreeks, priceCall]);

  if (error) return <div style={{ color: 'red' }}>Error loading WASM: {error.message}</div>;
  if (loading || !isInitialized) return <div>Initializing WASM pricing engine...</div>;

  return (
    <div className="options-chain" style={{ padding: '20px', fontFamily: 'monospace' }}>
      <h2>Options Chain (Powered by WebAssembly ⚡)</h2>
      <div style={{ marginBottom: '10px', fontSize: '0.9em', color: '#666' }}>
        Rendered {options.length} options in <strong>{benchTime.toFixed(4)}ms</strong>
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ textAlign: 'left', borderBottom: '2px solid #333' }}>
            <th style={{ padding: '8px' }}>Strike</th>
            <th style={{ padding: '8px' }}>Price</th>
            <th style={{ padding: '8px' }}>Delta</th>
            <th style={{ padding: '8px' }}>Gamma</th>
            <th style={{ padding: '8px' }}>Vega</th>
            <th style={{ padding: '8px' }}>Theta</th>
          </tr>
        </thead>
        <tbody>
          {options.map((opt) => (
            <tr key={opt.strike} style={{ borderBottom: '1px solid #eee' }}>
              <td style={{ padding: '8px' }}>{opt.strike}</td>
              <td style={{ padding: '8px' }}>${opt.price.toFixed(2)}</td>
              <td style={{ padding: '8px' }}>{opt.greeks.delta.toFixed(4)}</td>
              <td style={{ padding: '8px' }}>{opt.greeks.gamma.toFixed(4)}</td>
              <td style={{ padding: '8px' }}>{opt.greeks.vega.toFixed(4)}</td>
              <td style={{ padding: '8px' }}>{opt.greeks.theta.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
