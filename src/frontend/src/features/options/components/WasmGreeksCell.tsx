import React from 'react';
import { IconButton, Tooltip, Stack, Typography } from '@mui/material';
import { ShowChart } from '@mui/icons-material';

interface WasmGreeksCellProps {
  greeks?: {
    delta: number;
    gamma: number;
    vega: number;
    theta: number;
    rho: number;
  };
  price?: number;
}

// ⚡ Bolt: Performance Optimization
// Removed individual async WebWorker calls from this component.
// Now accepts pre-calculated greeks and price as props.
// This prevents triggering N WebWorker calls when rendering a large DataGrid list,
// which previously caused massive overhead and slow rendering times.
export const WasmGreeksCell = React.memo(({
  greeks,
  price,
}: WasmGreeksCellProps) => {

  if (!greeks || typeof price === 'undefined') {
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
          <Typography variant="caption">Delta: {delta?.toFixed(4) || '---'}</Typography>
          <Typography variant="caption">Gamma: {gamma?.toFixed(4) || '---'}</Typography>
          <Typography variant="caption">Vega: {vega?.toFixed(4) || '---'}</Typography>
          <Typography variant="caption">Theta: {theta?.toFixed(4) || '---'}</Typography>
          <Typography variant="caption">Rho: {rho?.toFixed(4) || '---'}</Typography>
          <Typography variant="caption">Theor. Price: ${price?.toFixed(4) || '---'}</Typography>
        </Stack>
      }
    >
      <IconButton size="small" color="primary" aria-label="View Greeks details">
        <ShowChart fontSize="small" />
      </IconButton>
    </Tooltip>
  );
});
