import { Box, CircularProgress, Typography, useTheme, alpha } from '@mui/material';
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
  put_delta: number;
  put_gamma: number;
  put_iv: number;
  underlying_price: number;
}

export const GreeksHeatmap: React.FC<GreeksHeatmapProps> = React.memo(({ symbol, greek }) => {
  const theme = useTheme();
  const { batchCalculate, isLoaded: isWasmLoaded } = useWasmPricing();

  const { data: gqlData, loading: isLoading, error } = useQuery(GET_OPTIONS_FOR_HEATMAP, {
    variables: { symbol },
  });

  // Transform flat GraphQL nodes into aggregated OptionData
  const optionsData = useMemo(() => {
    if (!gqlData?.options?.edges) return [];

    const nodes = gqlData.options.edges.map((e: any) => e.node);
    const spot = gqlData.marketData?.lastPrice || 155.0;
    const groups: Record<string, OptionData> = {};

    nodes.forEach((node: any) => {
      const key = `${node.strike}-${node.expiry}`;
      if (!groups[key]) {
        groups[key] = {
          strike: node.strike,
          expiry: node.expiry,
          underlying_price: spot,
          call_delta: 0, call_gamma: 0, call_iv: 0,
          put_delta: 0, put_gamma: 0, put_iv: 0,
        };
      }

      const isCall = node.optionType.toUpperCase() === 'CALL';
      const prefix = isCall ? 'call_' : 'put_';

      const target = groups[key] as any;
      target[`${prefix}iv`] = node.iv;
      target[`${prefix}delta`] = node.delta;
      target[`${prefix}gamma`] = node.gamma;
    });

    return Object.values(groups);
  }, [gqlData]);

  // Enrich data with WASM for Greeks if loaded
  const processedData = useMemo(() => {
    if (!optionsData || !isWasmLoaded) return optionsData || [];

    // Generate all params for batch calculation
    const params = optionsData.map(d => ({
      spot: d.underlying_price,
      strike: d.strike,
      time: 30 / 365, // Mock time
      vol: d.call_iv,
      rate: 0.05,
      div: 0.0,
      is_call: true
    }));

    // @ts-ignore
    const results = batchCalculate(params);

    return optionsData.map((d, i) => {
      // @ts-ignore
      const result = results[i];
      if (result) {
        return {
          ...d,
          // @ts-ignore
          call_delta: result.greeks.delta,
          // @ts-ignore
          call_gamma: result.greeks.gamma,
          // @ts-ignore
          call_vega: result.greeks.vega,
          // @ts-ignore
          call_theta: result.greeks.theta,
        };
      }
      return d;
    });
  }, [optionsData, isWasmLoaded, batchCalculate]);

  const chartOptions = useMemo(() => {
    if (!processedData || processedData.length === 0) return null;

    const strikes = Array.from(new Set(processedData.map((d: OptionData) => d.strike))).sort((a: number, b: number) => a - b);
    const expiries = Array.from(new Set(processedData.map((d: OptionData) => d.expiry))).sort();

    const data = processedData.map((d: OptionData) => {
      const strikeIdx = strikes.indexOf(d.strike);
      const expiryIdx = expiries.indexOf(d.expiry);

      let value = 0;
      if (greek === 'delta') value = d.call_delta;
      else if (greek === 'gamma') value = d.call_gamma;
      else if (greek === 'iv') value = d.call_iv;
      else if (greek === 'theta') value = (d as any).call_theta || 0;
      else if (greek === 'vega') value = (d as any).call_vega || 0;

      return [strikeIdx, expiryIdx, value];
    });

    return {
      tooltip: {
        position: 'top',
        formatter: (params: { data: [number, number, number] }) => {
          return `Strike: $${strikes[params.data[0]]}<br/>Expiry: ${expiries[params.data[1]]}<br/>${greek.toUpperCase()}: ${params.data[2].toFixed(4)}`;
        }
      },
      grid: {
        height: '80%',
        top: '10%',
        left: '10%',
        right: '5%'
      },
      xAxis: {
        type: 'category',
        data: strikes.map(s => `$${s}`),
        splitArea: { show: true },
        axisLabel: { color: theme.palette.text.secondary }
      },
      yAxis: {
        type: 'category',
        data: expiries,
        splitArea: { show: true },
        axisLabel: { color: theme.palette.text.secondary }
      },
      visualMap: {
        min: 0,
        max: 1,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '0%',
        inRange: {
          color: [
            alpha(theme.palette.primary.main, 0.1),
            theme.palette.primary.main,
            theme.palette.secondary.main
          ]
        },
        textStyle: { color: theme.palette.text.secondary }
      },
      series: [{
        name: `${greek.toUpperCase()} Heatmap`,
        type: 'heatmap',
        data: data,
        label: { show: false },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        }
      }],
      backgroundColor: 'transparent'
    };
  }, [processedData, greek, theme]);

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', minHeight: 300 }}>
        <CircularProgress aria-label="Loading Greeks heatmap" />
      </Box>
    );
  }

  if (error || !optionsData) {
    return (
      <Box sx={{ p: 4, textAlign: 'center' }}>
        <Typography color="error">Error loading heatmap data</Typography>
      </Box>
    );
  }

  return (
    <Box data-testid="greeks-heatmap-container" sx={{ width: '100%', height: '100%', minHeight: 400 }}>
      {chartOptions && (
        <ReactECharts
          echarts={echarts}
          option={chartOptions}
          style={{ height: '100%', width: '100%' }}
          theme={theme.palette.mode === 'dark' ? 'dark' : undefined}
        />
      )}
    </Box>
  );
});
