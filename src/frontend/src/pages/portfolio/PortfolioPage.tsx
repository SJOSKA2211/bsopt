import React from 'react';
import { Container, Grid, Paper, Typography } from '@mui/material';
import { PortfolioSummary } from '../../features/portfolio/components/PortfolioSummary';
import { PositionsSummary } from '../../features/portfolio/components/PositionsSummary';

export const PortfolioPage: React.FC = () => {
  return (
    <Container maxWidth="xl" sx={{ mt: 2 }}>
      <Typography variant="h4" gutterBottom fontWeight="bold">Portfolio</Typography>
      <Grid container spacing={3}>
        <Grid size={{ xs: 12, lg: 4 }}>
          <Paper sx={{ p: 0, height: 400, overflow: 'hidden' }}>
            <PortfolioSummary />
          </Paper>
        </Grid>
        <Grid size={{ xs: 12, lg: 8 }}>
          <Paper sx={{ p: 0, height: 800, overflow: 'hidden' }}>
            <PositionsSummary />
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default PortfolioPage;
