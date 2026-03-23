import React from 'react';
import { IconButton, Tooltip, Stack, Typography } from '@mui/material';
import { ShowChart } from '@mui/icons-material';

interface WasmGreeksCellProps {
  price?: number;
  greeks?: {
    delta?: number;
    gamma?: number;
    vega?: number;
    theta?: number;
    rho?: number;
  };
  isLoading?: boolean;
}

export const WasmGreeksCell = React.memo(({
  price,
  greeks,
  isLoading
}: WasmGreeksCellProps) => {

  if (isLoading || price === undefined || !greeks) {
    return (
      <IconButton size="small" disabled aria-label="Greeks calculation pending">
        <ShowChart fontSize="small" color="disabled" />
      </IconButton>
    );
  }

  const { delta, gamma, vega, theta, rho } = greeks;

  return (
    <Tooltip
      title={
        <Stack spacing={0.5}>
          <Typography variant="subtitle2">Client-Side Greeks (WASM)</Typography>
          <Typography variant="caption">Delta: {delta !== undefined ? delta.toFixed(4) : 'N/A'}</Typography>
          <Typography variant="caption">Gamma: {gamma !== undefined ? gamma.toFixed(4) : 'N/A'}</Typography>
          <Typography variant="caption">Vega: {vega !== undefined ? vega.toFixed(4) : 'N/A'}</Typography>
          <Typography variant="caption">Theta: {theta !== undefined ? theta.toFixed(4) : 'N/A'}</Typography>
          <Typography variant="caption">Rho: {rho !== undefined ? rho.toFixed(4) : 'N/A'}</Typography>
          <Typography variant="caption">Theor. Price: ${price.toFixed(4)}</Typography>
        </Stack>
      }
    >
      <IconButton size="small" color="primary" aria-label="View Greeks details">
        <ShowChart fontSize="small" />
      </IconButton>
    </Tooltip>
  );
});
