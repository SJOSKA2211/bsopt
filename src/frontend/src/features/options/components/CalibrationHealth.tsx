import React from 'react';
import { Box, Paper, Typography, Grid, LinearProgress, Tooltip, IconButton } from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';

interface CalibrationHealthProps {
  metrics: {
    rmse: number;
    r_squared: number;
    feller_margin: number;
    last_update: string;
  };
}

export const CalibrationHealth: React.FC<CalibrationHealthProps> = ({ metrics }) => {
  const getStatusColor = (val: number, threshold: number, higherIsBetter = false) => {
    if (higherIsBetter) {
      return val > threshold ? 'success.main' : 'error.main';
    }
    return val < threshold ? 'success.main' : 'error.main';
  };

  return (
    <Paper sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" sx={{ flexGrow: 1 }}>Calibration Engine Health</Typography>
        <Tooltip title="Real-time monitoring of Heston/SVI model fitting quality.">
          <IconButton size="small"><InfoIcon fontSize="small" /></IconButton>
        </Tooltip>
      </Box>

      <Grid container spacing={2}>
        <Grid size={{ xs: 12 }}>
          <Box sx={{ mb: 1 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2">RMSE (Error)</Typography>
              <Typography variant="body2" color={getStatusColor(metrics.rmse, 0.05)}>{(metrics.rmse * 100).toFixed(2)}%</Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={Math.min(100, metrics.rmse * 1000)} 
              color={metrics.rmse < 0.05 ? "success" : "error"}
            />
          </Box>
        </Grid>

        <Grid size={{ xs: 6 }}>
          <Typography variant="caption" color="text.secondary">R-Squared</Typography>
          <Typography variant="body1" color={getStatusColor(metrics.r_squared, 0.95, true)}>
            {metrics.r_squared.toFixed(4)}
          </Typography>
        </Grid>

        <Grid size={{ xs: 6 }}>
          <Typography variant="caption" color="text.secondary">Feller Margin</Typography>
          <Typography variant="body1" color={metrics.feller_margin > 0 ? "success.main" : "error.main"}>
            {metrics.feller_margin.toFixed(4)}
          </Typography>
        </Grid>

        <Grid size={{ xs: 12 }}>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', textAlign: 'right', mt: 1 }}>
            Last Recalibration: {new Date(metrics.last_update).toLocaleTimeString()}
          </Typography>
        </Grid>
      </Grid>
    </Paper>
  );
};
