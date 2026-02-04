import React, { useMemo } from 'react';
import { IconButton, Tooltip, Stack, Typography } from '@mui/material';
import { ShowChart } from '@mui/icons-material';
import { useWasmPricing } from '../../../hooks/useWasmPricing';

interface WasmGreeksCellProps {
  spot: number;
  strike: number;
  time: number; // Time to maturity in years
  vol: number;
  rate: number;
  div: number;
  isCall: boolean;
}

export const WasmGreeksCell: React.FC<WasmGreeksCellProps> = ({
  spot,
  strike,
  time,
  vol,
  rate,
  div,
  isCall,
}) => {
  const { priceOption, isLoaded } = useWasmPricing();

  const result = useMemo(() => {
    if (!isLoaded) return null;
    return priceOption({
      spot,
      strike,
      time,
      vol,
      rate,
      div,
      is_call: isCall,
    });
  }, [isLoaded, spot, strike, time, vol, rate, div, isCall, priceOption]);

  if (!isLoaded || !result) {
    return (
      <IconButton size="small" disabled>
        <ShowChart fontSize="small" color="disabled" />
      </IconButton>
    );
  }

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
          <Typography variant="caption">Theor. Price: ${result.price.toFixed(4)}</Typography>
        </Stack>
      }
    >
      <IconButton size="small" color="primary">
        <ShowChart fontSize="small" />
      </IconButton>
    </Tooltip>
  );
};
