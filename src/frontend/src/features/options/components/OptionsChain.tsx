import React, { useState, useMemo, useEffect } from 'react';
import {
  Box,
  Typography,
  Stack,
  Chip,
  TextField,
  InputAdornment,
  ToggleButtonGroup,
  ToggleButton,
  Tooltip,
  alpha,
  useTheme,
} from '@mui/material';
import {
  DataGrid,
} from '@mui/x-data-grid';
import type {
  GridColDef,
  GridRenderCellParams,
  GridRowParams,
} from '@mui/x-data-grid';
import {
  Search,
  TrendingUp,
  TrendingDown,
} from '@mui/icons-material';
import { Zap } from '../../../components/common/Icons';
import { useQuery } from '@apollo/client/react';
import { gql } from '@apollo/client';
import { motion, AnimatePresence } from 'framer-motion';

// Custom components
import { QuickTradeButton } from './QuickTradeButton';
import { WasmGreeksCell } from './WasmGreeksCell';
import { useWasmPricing } from '../../../hooks/useWasmPricing';
import { usePricingStore } from '../../../store/usePricingStore';

// Types
export interface OptionChainRow {
  id: string;
  strike: number;
  expiry: string;
  call_bid: number;
  call_ask: number;
  call_last: number;
  call_volume: number;
  call_oi: number;
  call_iv: number;
  call_delta: number;
  call_gamma: number;
  put_bid: number;
  put_ask: number;
  put_last: number;
  put_volume: number;
  put_oi: number;
  put_iv: number;
  put_delta: number;
  put_gamma: number;
  underlying_price: number;
  call_theor?: number;
  put_theor?: number;
}

interface OptionNode {
  id: string;
  strike: number;
  expiry: string;
  optionType: string;
  bid: number;
  ask: number;
  lastPrice: number;
  volume: number;
  openInterest: number;
  iv: number;
  price: number;
  delta: number;
  gamma: number;
}

interface GqlData {
  marketData?: { lastPrice: number };
  options?: { edges: { node: OptionNode }[] };
}

interface WasmPricingResult {
  delta: number;
  gamma: number;
  price: number;
  iv: number;
  greeks?: {
    delta: number;
    gamma: number;
  };
}

const GET_OPTIONS_CHAIN = gql`
  query GetOptionsChain($symbol: String!, $expiryBucket: String) {
    marketData(symbol: $symbol) {
      lastPrice
    }
    options(underlying: $symbol, expiryBucket: $expiryBucket) {
      edges {
        node {
          id
          strike
          expiry
          optionType
          bid
          ask
          lastPrice
          volume
          openInterest
          iv
          price
          delta
          gamma
        }
      }
    }
  }
`;

interface OptionsChainProps {
  symbol: string;
  onOptionSelect?: (option: OptionChainRow) => void;
}

export const OptionsChain = React.memo(({ symbol, onOptionSelect }: OptionsChainProps) => {
  const theme = useTheme();
  const qfd = (theme.palette as unknown as { financial?: { qfd?: Record<string, string> } }).financial?.qfd;
  const [searchTerm, setSearchTerm] = useState('');
  const [expiryFilter, setExpiryFilter] = useState<string>('all');
  const [pricingModel, setModel] = useState<string>('black_scholes');
  const { 
    isLoaded: isWasmLoaded, 
    batchCalculate, 
    priceMonteCarlo, 
    priceAmerican, 
    priceHeston,
    batchPriceMonteCarlo,
    batchPriceAmerican,
    batchPriceHeston 
  } = useWasmPricing();
  const [enrichedResults, setEnrichedResults] = useState<WasmPricingResult[]>([]);
  const [lastSpot, setLastSpot] = useState<number>(0);

  // Fetch options chain data via Federated GraphQL
  const { data: gqlData, loading: isLoading } = useQuery(GET_OPTIONS_CHAIN, {
    variables: { symbol, expiryBucket: expiryFilter },
    pollInterval: 10000,
  });

  // Subscribe to real-time spot updates
  const priceData = usePricingStore((state: any) => state.prices[symbol]);
  const tick = priceData ? { lastPrice: priceData.price } : null;

  useEffect(() => {
    const newSpot = tick?.lastPrice || (gqlData as GqlData)?.marketData?.lastPrice;
    if (newSpot && newSpot !== lastSpot) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setLastSpot(newSpot);
    }
  }, [tick, gqlData, lastSpot]);

  // Transform flat GraphQL nodes into aggregated rows (grouped by strike and expiry)
  const optionsData = useMemo(() => {
    if (!(gqlData as GqlData)?.options?.edges) return [];

    const nodes: OptionNode[] = ((gqlData as GqlData).options?.edges || []).map((e: { node: OptionNode }) => e.node);
    const spot = lastSpot || 155.0;
    const groups: Record<string, OptionChainRow> = {};

    nodes.forEach((node: OptionNode) => {
      const key = `${node.strike}-${node.expiry}`;
      if (!groups[key]) {
        groups[key] = {
          id: key,
          strike: node.strike,
          expiry: node.expiry,
          underlying_price: spot,
          call_bid: 0, call_ask: 0, call_last: 0, call_volume: 0, call_oi: 0, call_iv: 0, call_delta: 0, call_gamma: 0,
          put_bid: 0, put_ask: 0, put_last: 0, put_volume: 0, put_oi: 0, put_iv: 0, put_delta: 0, put_gamma: 0,
        };
      }

      const isCall = node.optionType.toUpperCase() === 'CALL';
      const prefix = isCall ? 'call_' : 'put_';

      const item = groups[key] as unknown as Record<string, string | number | undefined>;
      item[`${prefix}bid`] = node.bid;
      item[`${prefix}ask`] = node.ask;
      item[`${prefix}last`] = node.lastPrice;
      item[`${prefix}volume`] = node.volume;
      item[`${prefix}oi`] = node.openInterest;
      item[`${prefix}iv`] = node.iv;
      item[`${prefix}delta`] = node.delta;
      item[`${prefix}gamma`] = node.gamma;
      item[`${prefix}theor`] = node.price;
    });

    return Object.values(groups);
  }, [gqlData, lastSpot]);

  // Handle WASM enrichment in an effect
  useEffect(() => {
    if (!optionsData.length || !isWasmLoaded) return;

    const runWasmEnrichment = async () => {
      // PROD-CHECK: Shared proxy parameters consistent with PositionsSummary
      const riskFreeRate = 0.045;
      const dividendYield = 0.015;
      const now = new Date();

      const allParams = [
        ...optionsData.map((row: OptionChainRow) => {
          const expiryDate = new Date(row.expiry);
          const timeToExpiry = Math.max(0.001, (expiryDate.getTime() - now.getTime()) / (1000 * 60 * 60 * 24 * 365));
          return {
            spot: row.underlying_price,
            strike: row.strike,
            time: timeToExpiry,
            vol: row.call_iv || 0.25,
            rate: riskFreeRate,
            div: dividendYield,
            is_call: true
          };
        }),
        ...optionsData.map((row: OptionChainRow) => {
          const expiryDate = new Date(row.expiry);
          const timeToExpiry = Math.max(0.001, (expiryDate.getTime() - now.getTime()) / (1000 * 60 * 60 * 24 * 365));
          return {
            spot: row.underlying_price,
            strike: row.strike,
            time: timeToExpiry,
            vol: row.put_iv || 0.25,
            rate: riskFreeRate,
            div: dividendYield,
            is_call: false
          };
        })
      ];

      let results: WasmPricingResult[] | undefined;
      try {
        if (pricingModel === 'black_scholes') {
          const raw = await batchCalculate(allParams);
          results = raw.map((r: any) => ({
            price: r.price,
            delta: r.greeks.delta,
            gamma: r.greeks.gamma,
            iv: 0, 
            greeks: r.greeks
          }));
        } else if (pricingModel === 'monte_carlo') {
          const flatParams = allParams.flatMap(p => [p.spot, p.strike, p.time, p.vol, p.rate, p.div, p.is_call ? 1 : 0]);
          const prices = await (batchPriceMonteCarlo as (p: number[], n?: number) => Promise<Float64Array>)(flatParams, 10000);
          results = Array.from(prices).map(p => ({ price: p, delta: 0, gamma: 0, iv: 0 }));
        } else if (pricingModel === 'crank_nicolson') {
          const flatParams = allParams.flatMap(p => [p.spot, p.strike, p.time, p.vol, p.rate, p.div, p.is_call ? 1 : 0]);
          const prices = await (batchPriceAmerican as (p: number[], m?: number, n?: number) => Promise<Float64Array>)(flatParams, 200, 200);
          results = Array.from(prices).map(p => ({ price: p, delta: 0, gamma: 0, iv: 0 }));
        } else if (pricingModel === 'heston') {
          const flatParams = allParams.flatMap(p => [
            p.spot, p.strike, p.time, p.rate, 0.04, 2.0, 0.04, 0.3, -0.7 // spot, strike, time, r, v0, kappa, theta, sigma, rho
          ]);
          const prices = await (batchPriceHeston as (p: number[]) => Promise<Float64Array>)(flatParams);
          results = Array.from(prices).map(p => ({ price: p, delta: 0, gamma: 0, iv: 0 }));
        }
      } catch (e: unknown) {
        console.error('WASM Batch enrichment failed:', e);
        return;
      }

      if (results) {
        setEnrichedResults(results);
      }
    };

    runWasmEnrichment();
  }, [optionsData, isWasmLoaded, pricingModel, batchCalculate, priceMonteCarlo, priceAmerican, priceHeston]);

  // Filter, sort and enrich data
  const processedData = useMemo(() => {
    if (!optionsData) return [];

    let filtered = optionsData;

    if (searchTerm) {
      const search = searchTerm.toLowerCase();
      filtered = filtered.filter((row: OptionChainRow) =>
        row.strike.toString().includes(search)
      );
    }

    if (!isWasmLoaded || enrichedResults.length === 0) return filtered;

    const half = filtered.length;
    return filtered.map((row: OptionChainRow, i: number) => ({
      ...row,
      call_theor: enrichedResults[i]?.price,
      call_delta: enrichedResults[i]?.greeks?.delta ?? row.call_delta,
      call_gamma: enrichedResults[i]?.greeks?.gamma ?? row.call_gamma,
      put_theor: enrichedResults[i + half]?.price,
      put_delta: enrichedResults[i + half]?.greeks?.delta ?? row.put_delta,
      put_gamma: enrichedResults[i + half]?.greeks?.gamma ?? row.put_gamma,
    }));
  }, [optionsData, searchTerm, isWasmLoaded, enrichedResults]);

  const handleModelChange = React.useCallback((_: React.MouseEvent<HTMLElement> | null, value: string | null) => {
    if (value) setModel(value);
  }, []);

  const handleSearchChange = React.useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
  }, []);

  const handleExpiryChange = React.useCallback((_: React.MouseEvent<HTMLElement> | null, value: string | null) => {
    if (value) setExpiryFilter(value);
  }, []);

  const handleRowClick = React.useCallback((params: GridRowParams) => {
    onOptionSelect?.(params.row as OptionChainRow);
  }, [onOptionSelect]);

  // Column definitions
  const columns: GridColDef[] = useMemo(() => [
    // CALL OPTIONS
    {
      field: 'call_theor',
      headerName: 'Model',
      width: 80,
      headerClassName: 'call-header',
      renderCell: (params: GridRenderCellParams) => (
        <Typography
          variant="price"
          sx={{ fontStyle: 'italic', color: alpha(theme.palette.text.secondary, 0.7), fontSize: '0.8rem' }}
        >
          ${params.value?.toFixed(2) || '---'}
        </Typography>
      ),
    },
    {
      field: 'call_bid',
      headerName: 'Bid',
      width: 80,
      headerClassName: 'call-header',
      renderCell: (params: GridRenderCellParams) => (
        <Typography
          variant="price"
          sx={{ fontWeight: 'bold', color: theme.palette.success.main }}
        >
          ${params.value?.toFixed(2)}
        </Typography>
      ),
    },
    {
      field: 'call_ask',
      headerName: 'Ask',
      width: 80,
      headerClassName: 'call-header',
      renderCell: (params: GridRenderCellParams) => (
        <Typography
          variant="price"
          sx={{ fontWeight: 'bold', color: theme.palette.error.main }}
        >
          ${params.value?.toFixed(2)}
        </Typography>
      ),
    },
    {
      field: 'call_last',
      headerName: 'Last',
      width: 90,
      headerClassName: 'call-header',
      renderCell: (params: GridRenderCellParams) => {
        const row = params.row as OptionChainRow;
        const change = row.call_last - row.call_bid;
        const percentChange = (change / row.call_bid) * 100;

        return (
          <Stack spacing={0}>
            <Typography variant="price" sx={{ fontWeight: 800 }}>
              ${params.value?.toFixed(2)}
            </Typography>
            <Typography
              variant="caption"
              sx={{
                color: percentChange >= 0 ? 'success.main' : 'error.main',
                fontSize: '0.65rem',
                fontWeight: 800
              }}
            >
              {percentChange >= 0 ? '+' : ''}{percentChange.toFixed(1)}%
            </Typography>
          </Stack>
        );
      },
    },
    {
      field: 'call_volume',
      headerName: 'Vol',
      width: 70,
      headerClassName: 'call-header',
      renderCell: (params: GridRenderCellParams) => {
        const row = params.row as OptionChainRow;
        const isHot = row.call_volume > row.call_oi * 1.5 && row.call_volume > 100;
        return (
          <Stack direction="row" spacing={0.5} alignItems="center">
            <Typography variant="caption" sx={{ fontWeight: 700, opacity: 0.8, color: isHot ? 'warning.main' : 'inherit' }}>
              {params.value?.toLocaleString()}
            </Typography>
            {isHot && (
              <Box
                component={motion.div}
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ repeat: Infinity, duration: 1.5 }}
              >
                <TrendingUp sx={{ fontSize: 12, color: 'warning.main' }} />
              </Box>
            )}
          </Stack>
        );
      }
    },
    {
      field: 'call_oi',
      headerName: 'OI',
      width: 70,
      headerClassName: 'call-header',
      renderCell: (params: GridRenderCellParams) => (
        <Typography variant="caption" sx={{ fontWeight: 700, opacity: 0.8 }}>
          {params.value?.toLocaleString()}
        </Typography>
      )
    },
    {
      field: 'call_iv',
      headerName: 'IV',
      width: 70,
      headerClassName: 'call-header',
      renderCell: (params: GridRenderCellParams) => (
        <Tooltip title="Implied Volatility">
          <Typography variant="percentage" sx={{ color: qfd?.electrum, opacity: 0.9 }}>
            {(params.value * 100).toFixed(1)}%
          </Typography>
        </Tooltip>
      ),
    },
    {
      field: 'call_greeks',
      headerName: 'Greeks',
      width: 100,
      headerClassName: 'call-header',
      renderCell: (params: GridRenderCellParams) => {
        const row = params.row as OptionChainRow;
        const now = new Date();
        const expiryDate = new Date(row.expiry);
        const timeToExpiry = Math.max(0.001, (expiryDate.getTime() - now.getTime()) / (1000 * 60 * 60 * 24 * 365));
        const rate = 0.045;
        const div = 0.015;

        return (
          <WasmGreeksCell
            spot={row.underlying_price}
            strike={row.strike}
            time={timeToExpiry}
            vol={row.call_iv || 0.25}
            rate={rate}
            div={div}
            isCall={true}
          />
        );
      },
    },
    {
      field: 'call_action',
      headerName: ' ',
      width: 60,
      headerClassName: 'call-header',
      renderCell: (params: GridRenderCellParams) => (
        <QuickTradeButton
          option={params.row}
          type="call"
          action="buy"
        />
      ),
    },

    // STRIKE COLUMN (CENTER)
    {
      field: 'strike',
      headerName: 'Strike',
      width: 120,
      headerClassName: 'strike-header',
      renderCell: (params: GridRenderCellParams) => {
        const row = params.row as OptionChainRow;
        const isATM = Math.abs(row.strike - row.underlying_price) < 1;
        const isITM_Call = row.strike < row.underlying_price;
        const isITM_Put = row.strike > row.underlying_price;

        return (
          <Box
            sx={{
              width: '100%',
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              position: 'relative',
              background: isATM
                ? `linear-gradient(90deg, ${alpha(qfd?.quantum ?? '#00FFFF', 0.05)} 0%, ${alpha(qfd?.quantum ?? '#00FFFF', 0.15)} 50%, ${alpha(qfd?.quantum ?? '#00FFFF', 0.05)} 100%)`
                : 'transparent',
              '&::before': {
                content: '""',
                position: 'absolute',
                left: 0,
                top: 0,
                bottom: 0,
                width: 4,
                background: isITM_Call ? `linear-gradient(to bottom, ${theme.palette.success.main}, ${alpha(theme.palette.success.main, 0.3)})` : 'transparent',
                borderRadius: '0 2px 2px 0'
              },
              '&::after': {
                content: '""',
                position: 'absolute',
                right: 0,
                top: 0,
                bottom: 0,
                width: 4,
                background: isITM_Put ? `linear-gradient(to bottom, ${theme.palette.error.main}, ${alpha(theme.palette.error.main, 0.3)})` : 'transparent',
                borderRadius: '2px 0 0 2px'
              }
            }}
          >
            <Typography
              variant="h6"
              sx={{
                fontWeight: 900,
                fontFamily: 'JetBrains Mono',
                color: isATM ? qfd?.quantum : 'text.primary',
                fontSize: '1rem',
                textShadow: isATM ? `0 0 10px ${alpha(qfd?.quantum ?? '#00FFFF', 0.5)}` : 'none'
              }}
            >
              ${params.value}
            </Typography>
          </Box>
        );
      },
    },

    // PUT OPTIONS
    {
      field: 'put_theor',
      headerName: 'Model',
      width: 80,
      headerClassName: 'put-header',
      renderCell: (params: GridRenderCellParams) => (
        <Typography
          variant="price"
          sx={{ fontStyle: 'italic', color: alpha(theme.palette.text.secondary, 0.7), fontSize: '0.8rem' }}
        >
          ${params.value?.toFixed(2) || '---'}
        </Typography>
      ),
    },
    {
      field: 'put_bid',
      headerName: 'Bid',
      width: 80,
      headerClassName: 'put-header',
      renderCell: (params: GridRenderCellParams) => (
        <Typography
          variant="price"
          sx={{ fontWeight: 'bold', color: theme.palette.success.main }}
        >
          ${params.value?.toFixed(2)}
        </Typography>
      ),
    },
    {
      field: 'put_ask',
      headerName: 'Ask',
      width: 80,
      headerClassName: 'put-header',
      renderCell: (params: GridRenderCellParams) => (
        <Typography
          variant="price"
          sx={{ fontWeight: 'bold', color: theme.palette.error.main }}
        >
          ${params.value?.toFixed(2)}
        </Typography>
      ),
    },
    {
      field: 'put_last',
      headerName: 'Last',
      width: 90,
      headerClassName: 'put-header',
      renderCell: (params: GridRenderCellParams) => {
        const row = params.row as OptionChainRow;
        const change = row.put_last - row.put_bid;
        const percentChange = (change / row.put_bid) * 100;

        return (
          <Stack spacing={0}>
            <Typography variant="price" sx={{ fontWeight: 800 }}>
              ${params.value?.toFixed(2)}
            </Typography>
            <Typography
              variant="caption"
              sx={{
                color: percentChange >= 0 ? 'success.main' : 'error.main',
                fontSize: '0.65rem',
                fontWeight: 800
              }}
            >
              {percentChange >= 0 ? '+' : ''}{percentChange.toFixed(1)}%
            </Typography>
          </Stack>
        );
      },
    },
    {
      field: 'put_volume',
      headerName: 'Vol',
      width: 70,
      headerClassName: 'put-header',
      renderCell: (params: GridRenderCellParams) => {
        const row = params.row as OptionChainRow;
        const isHot = row.put_volume > row.put_oi * 1.5 && row.put_volume > 100;
        return (
          <Stack direction="row" spacing={0.5} alignItems="center">
            <Typography variant="caption" sx={{ fontWeight: 700, opacity: 0.8, color: isHot ? 'warning.main' : 'inherit' }}>
              {params.value?.toLocaleString()}
            </Typography>
            {isHot && (
              <Box
                component={motion.div}
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ repeat: Infinity, duration: 1.5 }}
              >
                <TrendingUp sx={{ fontSize: 12, color: 'warning.main' }} />
              </Box>
            )}
          </Stack>
        );
      }
    },
    {
      field: 'put_oi',
      headerName: 'OI',
      width: 70,
      headerClassName: 'put-header',
      renderCell: (params: GridRenderCellParams) => (
        <Typography variant="caption" sx={{ fontWeight: 700, opacity: 0.8 }}>
          {params.value?.toLocaleString()}
        </Typography>
      )
    },
    {
      field: 'put_iv',
      headerName: 'IV',
      width: 70,
      headerClassName: 'put-header',
      renderCell: (params: GridRenderCellParams) => (
        <Tooltip title="Implied Volatility">
          <Typography variant="percentage" sx={{ color: qfd?.electrum, opacity: 0.9 }}>
            {(params.value * 100).toFixed(1)}%
          </Typography>
        </Tooltip>
      ),
    },
    {
      field: 'put_greeks',
      headerName: 'Greeks',
      width: 100,
      headerClassName: 'put-header',
      renderCell: (params: GridRenderCellParams) => {
        const row = params.row as OptionChainRow;
        const now = new Date();
        const expiryDate = new Date(row.expiry);
        const timeToExpiry = Math.max(0.001, (expiryDate.getTime() - now.getTime()) / (1000 * 60 * 60 * 24 * 365));
        const rate = 0.045;
        const div = 0.015;

        return (
          <WasmGreeksCell
            spot={row.underlying_price}
            strike={row.strike}
            time={timeToExpiry}
            vol={row.put_iv || 0.25}
            rate={rate}
            div={div}
            isCall={false}
          />
        );
      },
    },
    {
      field: 'put_action',
      headerName: ' ',
      width: 60,
      headerClassName: 'put-header',
      renderCell: (params: GridRenderCellParams) => (
        <QuickTradeButton
          option={params.row}
          type="put"
          action="buy"
        />
      ),
    },
  ], [theme, qfd]);

  return (
    <Box
      component={motion.div}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}
    >
      {/* Header with filters */}
      <Stack
        direction="row"
        spacing={2}
        alignItems="center"
        sx={{ p: 3, borderBottom: `0.5px solid ${alpha(theme.palette.divider, 0.1)}`, background: alpha(theme.palette.background.paper, 0.2) }}
      >
        <Stack direction="row" spacing={2} alignItems="center" sx={{ flexGrow: 1 }}>
          <Typography variant="h5" sx={{ fontWeight: 900, letterSpacing: '-0.02em', fontFamily: 'Outfit' }}>
            OptX Matrix: {symbol}
          </Typography>
          <AnimatePresence mode="wait">
            <Box
              key={lastSpot}
              component={motion.div}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10 }}
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                bgcolor: alpha(qfd?.quantum ?? '#00FFFF', 0.1),
                px: 1.5,
                py: 0.5,
                borderRadius: 2,
                border: `1px solid ${alpha(qfd?.quantum ?? '#00FFFF', 0.2)}`
              }}
            >
              <Typography
                variant="price"
                sx={{
                  fontWeight: 900,
                  color: qfd?.quantum,
                  fontSize: '0.9rem'
                }}
              >
                ${lastSpot.toFixed(2)}
              </Typography>
              {tick?.lastPrice && tick.lastPrice > ((gqlData as GqlData)?.marketData?.lastPrice || 0) ?
                <TrendingUp sx={{ fontSize: 14, color: 'success.main' }} /> :
                <TrendingDown sx={{ fontSize: 14, color: 'error.main' }} />
              }
            </Box>
          </AnimatePresence>
          {isWasmLoaded && (
            <Chip
              label="WASM SIMD"
              size="small"
              icon={<Zap size={12} />}
              sx={{
                height: 20,
                fontSize: '0.65rem',
                fontWeight: 900,
                bgcolor: alpha(qfd?.electrum ?? '#D4AF37', 0.1),
                color: qfd?.electrum,
                border: `1px solid ${alpha(qfd?.electrum ?? '#D4AF37', 0.2)}`
              }}
            />
          )}
        </Stack>

        <TextField
          size="small"
          placeholder="Filter strike..."
          value={searchTerm}
          onChange={handleSearchChange}
          inputProps={{
            'aria-label': 'Filter strike',
          }}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search sx={{ color: 'text.secondary', opacity: 0.5 }} />
              </InputAdornment>
            ),
          }}
          sx={{
            width: 180,
            '& .MuiOutlinedInput-root': {
              borderRadius: 3,
              bgcolor: alpha('#f8fafc', 0.05)
            }
          }}
        />

        <ToggleButtonGroup
          value={pricingModel}
          exclusive
          onChange={handleModelChange}
          aria-label="Select pricing model"
          size="small"
          sx={{
            bgcolor: alpha('#f8fafc', 0.05),
            borderRadius: 3,
            p: 0.5,
            '& .MuiToggleButton-root': {
              border: 'none',
              borderRadius: '8px !important',
              px: 1.5,
              fontSize: '0.7rem',
              fontWeight: 800
            }
          }}
        >
          <ToggleButton value="black_scholes">BS</ToggleButton>
          <ToggleButton value="monte_carlo">MC</ToggleButton>
          <ToggleButton value="crank_nicolson">CN</ToggleButton>
          <ToggleButton value="heston">HES</ToggleButton>
        </ToggleButtonGroup>

        <ToggleButtonGroup
          value={expiryFilter}
          exclusive
          onChange={handleExpiryChange}
          size="small"
          aria-label="Expiry Filter"
          sx={{
            bgcolor: alpha('#f8fafc', 0.05),
            borderRadius: 3,
            p: 0.5,
            '& .MuiToggleButton-root': {
              border: 'none',
              borderRadius: '8px !important',
              px: 1.5,
              fontSize: '0.7rem',
              fontWeight: 800
            }
          }}
        >
          <ToggleButton value="all">ALL</ToggleButton>
          <ToggleButton value="week">1W</ToggleButton>
          <ToggleButton value="month">1M</ToggleButton>
        </ToggleButtonGroup>
      </Stack>

      {/* Options Chain Grid */}
      <Box sx={{ flex: 1, position: 'relative' }}>
        <DataGrid
          rows={processedData}
          columns={columns}
          aria-label="Options Chain Data Grid"
          getRowId={(row: any) => row.id}
          loading={isLoading}
          disableRowSelectionOnClick
          onRowClick={handleRowClick}
          sx={{
            border: 'none',
            '& .MuiDataGrid-columnHeaders': {
              bgcolor: alpha(theme.palette.background.paper, 0.5),
              minHeight: '48px !important',
              borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`
            },
            '& .MuiDataGrid-columnHeaderTitle': {
              fontWeight: 900,
              fontSize: '0.75rem',
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: theme.palette.text.secondary
            },
            '& .call-header': {
              borderBottom: `2px solid ${alpha(theme.palette.success.main, 0.5)} !important`,
              '& .MuiDataGrid-columnHeaderTitle': { color: theme.palette.success.main, opacity: 0.8 }
            },
            '& .put-header': {
              borderBottom: `2px solid ${alpha(theme.palette.error.main, 0.5)} !important`,
              '& .MuiDataGrid-columnHeaderTitle': { color: theme.palette.error.main, opacity: 0.8 }
            },
            '& .strike-header': {
              borderBottom: `2px solid ${alpha(qfd?.quantum ?? '#00FFFF', 0.5)} !important`,
              '& .MuiDataGrid-columnHeaderTitle': { color: qfd?.quantum, opacity: 0.8 }
            },
            '& .MuiDataGrid-row': {
              minHeight: '56px !important',
              transition: 'background-color 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
              borderBottom: `1px solid ${alpha(theme.palette.divider, 0.05)}`
            },
            '& .MuiDataGrid-row:hover': {
              backgroundColor: alpha(qfd?.quantum ?? '#00FFFF', 0.04),
              boxShadow: `inset 0 0 20px ${alpha(qfd?.quantum ?? '#00FFFF', 0.08)}`,
              cursor: 'pointer',
            },
            '& .MuiDataGrid-cell': {
              border: 'none',
              display: 'flex',
              alignItems: 'center',
              px: 2
            }
          }}
          initialState={{
            pagination: { paginationModel: { pageSize: 20 } },
          }}
          pageSizeOptions={[20, 50, 100]}
        />
      </Box>
    </Box>
  );
});

export default OptionsChain;