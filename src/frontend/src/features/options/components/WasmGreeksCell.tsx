import React from 'react';
import { IconButton, Tooltip, Stack, Typography } from '@mui/material';
import { ShowChart } from '@mui/icons-material';

interface WasmGreeksCellProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  result?: any;
}

export const WasmGreeksCell = React.memo(({
  result,
}: WasmGreeksCellProps) => {

  if (!result) {
    return (
      <IconButton size="small" disabled aria-label="Greeks calculation pending">
        <ShowChart fontSize="small" color="disabled" />
      </IconButton>
    );
  }

  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore
  const { delta, gamma, vega, theta, rho } = result.greeks;

  return (
    <Tooltip
      title={
        <Stack spacing={0.5}>
          <Typography variant="subtitle2">Client-Side Greeks (WASM)</Typography>
          <Typography variant="caption">Delta: {delta.toFixed(4)}</Typography>
          <Typography variant="caption">Gamma: {gamma.toFixed(4)}</Typography>
          <Typography variant="caption">Vega: {vega.toFixed(4)}</Typography>
          <Typography variant="caption">Theta: {theta.toFixed(4)}</Typography>
          <Typography variant="caption">Rho: {rho.toFixed(4)}</Typography>
          {/* eslint-disable-next-line @typescript-eslint/ban-ts-comment */}
          {/* @ts-ignore */}
          <Typography variant="caption">Theor. Price: ${result.price.toFixed(4)}</Typography>
        </Stack>
      }
    >
      <IconButton size="small" color="primary" aria-label="View Greeks details">
        <ShowChart fontSize="small" />
      </IconButton>
    </Tooltip>
  );
});
