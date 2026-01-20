import React, { useState, useMemo } from 'react';
import {
  Box,
  Typography,
  Stack,
  Chip,
  IconButton,
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
  ShowChart,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';

// Custom components
import { QuickTradeButton } from './QuickTradeButton';

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
}

interface OptionsChainProps {
  symbol: string;
  onOptionSelect?: (option: OptionChainRow) => void;
}

export const OptionsChain: React.FC<OptionsChainProps> = React.memo(({ symbol, onOptionSelect }) => {
  const theme = useTheme();
  const [searchTerm, setSearchTerm] = useState('');
  const [expiryFilter, setExpiryFilter] = useState<string>('all');
  
  // Fetch options chain data
  const { data: optionsData, isLoading } = useQuery({
    queryKey: ['options-chain', symbol, expiryFilter],
    queryFn: async () => {
      const response = await fetch(
        `/api/v1/options/chain?symbol=${symbol}&expiry=${expiryFilter}`
      );
      return response.json();
    },
    refetchInterval: 3000, // Refresh every 3 seconds
  });
  
  // Filter and sort data
  const filteredData = useMemo(() => {
    if (!optionsData) return [];
    
    let filtered = optionsData;
    
    if (searchTerm) {
      const search = searchTerm.toLowerCase();
      filtered = filtered.filter((row: OptionChainRow) =>
        row.strike.toString().includes(search)
      );
    }
    
    return filtered;
  }, [optionsData, searchTerm]);

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
      renderCell: () => {
        return (
            <IconButton size="small">
              <ShowChart fontSize="small" />
            </IconButton>
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
    {
      field: 'put_greeks',
      headerName: 'Greeks',
      width: 100,
      headerClassName: 'put-header',
      renderCell: () => {
        return (
            <IconButton size="small">
              <ShowChart fontSize="small" />
            </IconButton>
        );
      },
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
      field: 'put_oi',
      headerName: 'OI',
      width: 80,
      headerClassName: 'put-header',
      valueFormatter: (value: number) =>
        value?.toLocaleString(),
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
          rows={filteredData}
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