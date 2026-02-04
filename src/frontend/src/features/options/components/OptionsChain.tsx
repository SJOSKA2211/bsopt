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
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';

// Custom components
import { QuickTradeButton } from './QuickTradeButton';
import { WasmGreeksCell } from './WasmGreeksCell';
import { useWasmPricing } from '../../../hooks/useWasmPricing';

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

interface OptionsChainProps {
  symbol: string;
  onOptionSelect?: (option: OptionChainRow) => void;
}

export const OptionsChain: React.FC<OptionsChainProps> = React.memo(({ symbol, onOptionSelect }) => {
  const theme = useTheme();
  const [searchTerm, setSearchTerm] = useState('');
  const [expiryFilter, setExpiryFilter] = useState<string>('all');
  const [pricingModel, setModel] = useState<string>('black_scholes');
  const { isLoaded: isWasmLoaded, batchCalculate, priceMonteCarlo, priceAmerican, priceHeston } = useWasmPricing();
  const [enrichedResults, setEnrichedResults] = useState<any[]>([]);
  
  // Fetch options chain data
  const { data: optionsData, isLoading } = useQuery({
    queryKey: ['options-chain', symbol, expiryFilter],
    queryFn: async () => {
      const response = await fetch(
        `/api/v1/options/chain?symbol=${symbol}&expiry=${expiryFilter}`
      );
      return response.json();
    },
    refetchInterval: 3000,
  });

  // Handle WASM enrichment in an effect to avoid useMemo anti-pattern with async
  useEffect(() => {
    if (!optionsData || !isWasmLoaded) return;

    const runWasmEnrichment = async () => {
      const rate = 0.05;
      const div = 0.0;
      const now = new Date();

      const allParams = [
        ...optionsData.map((row: OptionChainRow) => {
          const expiryDate = new Date(row.expiry);
          const timeToExpiry = Math.max(0.001, (expiryDate.getTime() - now.getTime()) / (1000 * 60 * 60 * 24 * 365));
          return {
            spot: row.underlying_price,
            strike: row.strike,
            time: timeToExpiry,
            vol: row.call_iv,
            rate,
            div,
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
            vol: row.put_iv,
            rate,
            div,
            is_call: false
          };
        })
      ];

      let results;
      if (pricingModel === 'black_scholes') {
        results = await batchCalculate(allParams);
      } else if (pricingModel === 'monte_carlo') {
        // Run MC for each row (SOTA: parallelized in worker)
        results = await Promise.all(allParams.map(p => priceMonteCarlo(p, 10000)));
      } else if (pricingModel === 'crank_nicolson') {
        results = await Promise.all(allParams.map(p => priceAmerican(p)));
      } else if (pricingModel === 'heston') {
        results = await Promise.all(allParams.map(p => priceHeston({...p, v0: 0.04, kappa: 2.0, theta: 0.04, sigma: 0.3, rho: -0.7})));
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
      put_theor: enrichedResults[i + half]?.price,
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
          sx={{ fontStyle: 'italic', color: 'text.secondary' }}
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
          color="financial.bid"
          sx={{ fontWeight: 'bold' }}
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
          color="financial.ask"
          sx={{ fontWeight: 'bold' }}
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
          <Stack spacing={0.5}>
            <Typography variant="price" sx={{ fontWeight: 'bold' }}>
              ${params.value?.toFixed(2)}
            </Typography>
            <Chip
              label={`${percentChange >= 0 ? '+' : ''}${percentChange.toFixed(1)}%`}
              size="small"
              color={percentChange >= 0 ? 'success' : 'error'}
              sx={{ height: 20, fontSize: '0.7rem' }}
            />
          </Stack>
        );
      },
    },
    {
      field: 'call_volume',
      headerName: 'Vol',
      width: 80,
      headerClassName: 'call-header',
      valueFormatter: (value: number) =>
        value?.toLocaleString(),
    },
    {
      field: 'call_oi',
      headerName: 'OI',
      width: 80,
      headerClassName: 'call-header',
      valueFormatter: (value: number) =>
        value?.toLocaleString(),
    },
    {
      field: 'call_iv',
      headerName: 'IV',
      width: 70,
      headerClassName: 'call-header',
      renderCell: (params: GridRenderCellParams) => (
        <Tooltip title="Implied Volatility">
          <Typography variant="percentage">
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
        // Mocking time/rate/div for demo purposes
        const timeToExpiry = 30 / 365; 
        const rate = 0.05;
        const div = 0.0;
        
        return (
          <WasmGreeksCell
            spot={row.underlying_price}
            strike={row.strike}
            time={timeToExpiry}
            vol={row.call_iv}
            rate={rate}
            div={div}
            isCall={true}
          />
        );
      },
    },
    {
      field: 'call_action',
      headerName: 'Action',
      width: 100,
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
              backgroundColor: isATM
                ? alpha(theme.palette.primary.main, 0.2)
                : 'transparent',
              borderLeft: isITM_Call ? `3px solid ${theme.palette.financial.positive}` : 'none',
              borderRight: isITM_Put ? `3px solid ${theme.palette.financial.positive}` : 'none',
            }}
          >
            <Typography
              variant="h6"
              fontWeight="bold"
              color={isATM ? 'primary.main' : 'text.primary'}
            >
              ${params.value}
            </Typography>
            {isATM && (
              <Chip
                label="ATM"
                size="small"
                color="primary"
                sx={{ ml: 1, height: 20 }}
              />
            )}
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
          sx={{ fontStyle: 'italic', color: 'text.secondary' }}
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
          color="financial.bid"
          sx={{ fontWeight: 'bold' }}
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
          color="financial.ask"
          sx={{ fontWeight: 'bold' }}
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
          <Stack spacing={0.5}>
            <Typography variant="price" sx={{ fontWeight: 'bold' }}>
              ${params.value?.toFixed(2)}
            </Typography>
            <Chip
              label={`${percentChange >= 0 ? '+' : ''}${percentChange.toFixed(1)}%`}
              size="small"
              color={percentChange >= 0 ? 'success' : 'error'}
              sx={{ height: 20, fontSize: '0.7rem' }}
            />
          </Stack>
        );
      },
    },
    {
      field: 'put_volume',
      headerName: 'Vol',
      width: 80,
      headerClassName: 'put-header',
      valueFormatter: (value: number) =>
        value?.toLocaleString(),
    },
    {
      field: 'put_oi',
      headerName: 'OI',
      width: 80,
      headerClassName: 'put-header',
      valueFormatter: (value: number) =>
        value?.toLocaleString(),
    },
    {
      field: 'put_iv',
      headerName: 'IV',
      width: 70,
      headerClassName: 'put-header',
      renderCell: (params: GridRenderCellParams) => (
        <Tooltip title="Implied Volatility">
          <Typography variant="percentage">
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
        const timeToExpiry = 30 / 365;
        const rate = 0.05;
        const div = 0.0;

        return (
          <WasmGreeksCell
            spot={row.underlying_price}
            strike={row.strike}
            time={timeToExpiry}
            vol={row.put_iv}
            rate={rate}
            div={div}
            isCall={false}
          />
        );
      },
    },
    {
      field: 'put_action',
      headerName: 'Action',
      width: 100,
      headerClassName: 'put-header',
      renderCell: (params: GridRenderCellParams) => (
        <QuickTradeButton
          option={params.row}
          type="put"
          action="buy"
        />
      ),
    },
  ], [theme]);
  
  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header with filters */}
      <Stack
        direction="row"
        spacing={2}
        alignItems="center"
        sx={{ p: 2, borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}` }}
      >
        <Typography variant="h6" sx={{ flexGrow: 1 }}>
          Options Chain - {symbol}
          {isWasmLoaded && (
            <Chip 
              label="WASM Engine Active" 
              size="small" 
              color="success" 
              variant="outlined" 
              sx={{ ml: 2, height: 20, fontSize: '0.65rem' }} 
            />
          )}
        </Typography>
        
        <TextField
          size="small"
          placeholder="Search strike..."
          value={searchTerm}
          onChange={handleSearchChange}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search />
              </InputAdornment>
            ),
          }}
          sx={{ width: 200 }}
        />
        
        <ToggleButtonGroup
          value={pricingModel}
          exclusive
          onChange={handleModelChange}
          size="small"
          sx={{ mr: 2 }}
        >
          <ToggleButton value="black_scholes">BS</ToggleButton>
          <ToggleButton value="monte_carlo">MC</ToggleButton>
          <ToggleButton value="crank_nicolson">CN</ToggleButton>
          <ToggleButton value="heston">Heston</ToggleButton>
        </ToggleButtonGroup>

        <ToggleButtonGroup
          value={expiryFilter}
          exclusive
          onChange={handleExpiryChange}
          size="small"
        >
          <ToggleButton value="all">All</ToggleButton>
          <ToggleButton value="week">1W</ToggleButton>
          <ToggleButton value="month">1M</ToggleButton>
          <ToggleButton value="quarter">3M</ToggleButton>
        </ToggleButtonGroup>
      </Stack>
      
      {/* Options Chain Grid */}
      <Box sx={{ flex: 1 }}>
        <DataGrid
          rows={processedData}
          columns={columns}
          loading={isLoading}
          disableRowSelectionOnClick
          onRowClick={handleRowClick}
          sx={{
            border: 'none',
            
            '& .call-header': {
              backgroundColor: alpha(theme.palette.financial.positive, 0.1),
              borderBottom: `2px solid ${theme.palette.financial.positive}`,
            },
            
            '& .put-header': {
              backgroundColor: alpha(theme.palette.financial.negative, 0.1),
              borderBottom: `2px solid ${theme.palette.financial.negative}`,
            },
            
            '& .strike-header': {
              backgroundColor: alpha(theme.palette.primary.main, 0.15),
              borderBottom: `2px solid ${theme.palette.primary.main}`,
            },
            
            '& .MuiDataGrid-cell': {
              borderRight: `1px solid ${alpha(theme.palette.divider, 0.5)}`,
            },
            
            '& .MuiDataGrid-row:hover': {
              backgroundColor: alpha(theme.palette.primary.main, 0.05),
              cursor: 'pointer',
            },
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