import React, { useMemo, useState, useEffect } from 'react';
import { Box, CircularProgress, Typography, useTheme, alpha } from '@mui/material';
import ReactECharts from 'echarts-for-react';
import * as echarts from 'echarts';
import { useQuery } from '@apollo/client/react';
import { gql } from '@apollo/client';
import { useWasmPricing } from '../../../hooks/useWasmPricing';

const GET_OPTIONS_FOR_HEATMAP = gql`
  query GetOptionsForHeatmap($symbol: String!) {
    marketData(symbol: $symbol) {
      lastPrice
    }
    options(underlying: $symbol) {
      edges {
        node {
          id
          strike
          expiry
          optionType
          iv
          delta
          gamma
          # Add other greeks if available in schema
        }
      }
    }
  }
`;

interface GreeksHeatmapProps {
  symbol: string;
  greek: 'delta' | 'gamma' | 'iv' | 'theta' | 'vega';
}

interface OptionData {
  strike: number;
  expiry: string;
  call_delta: number;
  call_gamma: number;
  call_iv: number;
  call_theta?: number;
  call_vega?: number;
  put_delta: number;
  put_gamma: number;
  put_iv: number;
  underlying_price: number;
}

interface GqlData {
  marketData?: { lastPrice: number };
  options?: { edges: { node: Record<string, unknown> }[] };
}

export const GreeksHeatmap: React.FC<GreeksHeatmapProps> = React.memo(({ symbol, greek }: GreeksHeatmapProps) => {
  const theme = useTheme();
  const { batchCalculate, isLoaded: isWasmLoaded } = useWasmPricing();

  const { data: gqlData, loading: isLoading, error } = useQuery(GET_OPTIONS_FOR_HEATMAP, {
    variables: { symbol },
  });

  const optionsData = useMemo(() => {
    if (!(gqlData as GqlData)?.options?.edges) return [];
    const nodes = (gqlData as GqlData).options?.edges.map((e: any) => e.node) || [];
    const spot = (gqlData as GqlData).marketData?.lastPrice || 155.0;
    const groups: Record<string, OptionData> = {};

    nodes.forEach((node: any) => {
      const strike = Number(node.strike);
      const expiry = String(node.expiry);
      const key = `${strike}-${expiry}`;
      if (!groups[key]) {
        groups[key] = {
          strike,
          expiry,
          underlying_price: spot,
          call_delta: 0, call_gamma: 0, call_iv: 0,
          put_delta: 0, put_gamma: 0, put_iv: 0,
        };
      }
      const isCall = String(node.optionType).toUpperCase() === 'CALL';
      const prefix = isCall ? 'call_' : 'put_';
      const target = groups[key] as any;
      target[`${prefix}iv`] = Number(node.iv);
      target[`${prefix}delta`] = Number(node.delta);
      target[`${prefix}gamma`] = Number(node.gamma);
    });
    return Object.values(groups);
  }, [gqlData]);

  const [processedData, setProcessedData] = useState<OptionData[]>([]);

  useEffect(() => {
    if (!optionsData || !isWasmLoaded) {
      if (processedData.length !== (optionsData?.length || 0)) {
        setProcessedData(optionsData || []);
      }
      return;
    }
    const runEnrichment = async () => {
      const now = new Date();
      const params = optionsData.map(d => ({
        spot: d.underlying_price,
        strike: d.strike,
        time: Math.max(0.001, (new Date(d.expiry).getTime() - now.getTime()) / (31536000000)),
        vol: d.call_iv || 0.25,
        rate: 0.045,
        div: 0.015,
        is_call: true
      }));
      const results = await batchCalculate(params);
      if (!results) return;
      setProcessedData(optionsData.map((d, i) => {
        const r = results[i];
        return r ? { ...d, call_delta: r.greeks.delta, call_gamma: r.greeks.gamma, call_vega: r.greeks.vega, call_theta: r.greeks.theta } : d;
      }));
    };
    runEnrichment();
  }, [optionsData, isWasmLoaded, batchCalculate]);

  const { strikes, expiries, data } = useMemo(() => {
    const s = Array.from(new Set(processedData.map((d: OptionData) => d.strike))).sort((a: number, b: number) => a - b);
    const e = Array.from(new Set(processedData.map((d: OptionData) => d.expiry))).sort();
    const d = processedData.map((opt: OptionData) => {
      const val = greek === 'delta' ? opt.call_delta : greek === 'gamma' ? opt.call_gamma : greek === 'iv' ? opt.call_iv : greek === 'theta' ? opt.call_theta || 0 : opt.call_vega || 0;
      return [s.indexOf(opt.strike), e.indexOf(opt.expiry), val];
    });
    return { strikes: s, expiries: e, data: d };
  }, [processedData, greek]);

  const chartOptions = useMemo(() => {
    if (data.length === 0) return null;
    const greekColors = (theme.palette as any).financial?.greeks?.[greek] || theme.palette.primary.main;
    return {
      tooltip: {
        position: 'top',
        backgroundColor: alpha(theme.palette.background.paper, 0.9),
        borderColor: alpha(greekColors, 0.3),
        borderWidth: 1,
        textStyle: { color: theme.palette.text.primary, fontFamily: 'JetBrains Mono', fontSize: 12 },
        formatter: (params: any) => {
          const val = params.data;
          return `
            <div style="padding: 4px;">
              <div style="color: ${theme.palette.text.secondary}; font-size: 10px; font-weight: 800; margin-bottom: 4px;">OPTION DETECTORS</div>
              <div style="margin-bottom: 2px;">STRIKE: <span style="color: ${theme.palette.text.primary}; font-weight: 700;">$${strikes[val[0]]}</span></div>
              <div style="margin-bottom: 4px;">EXPIRY: <span style="color: ${theme.palette.text.primary}; font-weight: 700;">${expiries[val[1]]}</span></div>
              <div style="border-top: 1px solid ${alpha(theme.palette.divider, 0.1)}; padding-top: 4px;">
                <span style="color: ${greekColors}; font-weight: 900;">${greek.toUpperCase()}: ${val[2].toFixed(4)}</span>
              </div>
            </div>
          `;
        }
      },
      grid: { height: '75%', top: '10%', left: '12%', right: '5%', bottom: '15%' },
      xAxis: {
        type: 'category',
        data: strikes.map(s => `$${s}`),
        axisLine: { lineStyle: { color: alpha(theme.palette.divider, 0.1) } },
        axisLabel: { color: theme.palette.text.secondary, fontFamily: 'JetBrains Mono', fontSize: 10 }
      },
      yAxis: {
        type: 'category',
        data: expiries,
        axisLine: { lineStyle: { color: alpha(theme.palette.divider, 0.1) } },
        axisLabel: { color: theme.palette.text.secondary, fontFamily: 'JetBrains Mono', fontSize: 10 }
      },
      visualMap: {
        min: greek === 'delta' ? 0 : greek === 'theta' ? -1 : 0,
        max: greek === 'delta' ? 1 : greek === 'gamma' ? 0.1 : 1,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '0%',
        inRange: { color: [alpha(greekColors, 0.05), alpha(greekColors, 0.4), greekColors] },
        textStyle: { color: theme.palette.text.secondary, fontFamily: 'JetBrains Mono', fontSize: 10 }
      },
      series: [{
        name: `${greek.toUpperCase()} Heatmap`,
        type: 'heatmap',
        data: data,
        label: { show: false },
        itemStyle: { borderColor: theme.palette.mode === 'dark' ? '#020617' : '#fff', borderWidth: 1 },
        emphasis: { itemStyle: { shadowBlur: 20, shadowColor: alpha(greekColors, 0.5), borderWidth: 2, borderColor: greekColors } }
      }],
      backgroundColor: 'transparent'
    };
  }, [data, greek, theme, strikes, expiries]);

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', minHeight: 400 }}>
        <CircularProgress size={30} aria-label="Loading Greeks heatmap" />
      </Box>
    );
  }

  if (error || !optionsData) {
    return (
      <Box sx={{ p: 4, textAlign: 'center', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Typography color="error" variant="caption" sx={{ fontWeight: 800 }}>ERROR SYNCHRONIZING MARKET MANIFOLD</Typography>
      </Box>
    );
  }

  return (
    <Box data-testid="greeks-heatmap-container" sx={{ width: '100%', height: '100%', minHeight: 450, position: 'relative', p: 1 }}>
      {chartOptions && (
        <ReactECharts
          echarts={echarts}
          option={chartOptions}
          style={{ height: '100%', width: '100%' }}
          theme={theme.palette.mode === 'dark' ? 'dark' : undefined}
          notMerge={true}
        />
      )}
    </Box>
  );
});
