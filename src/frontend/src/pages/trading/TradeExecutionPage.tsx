import React from 'react';
import { Box, Paper, alpha, Grid } from '@mui/material';
import { DepthOfMarket } from '../../features/trading/components/DepthOfMarket';
import { LevelIIQuotes } from '../../features/trading/components/LevelIIQuotes';
import { OrderTicket } from '../../features/trading/components/OrderTicket';
import { TimeAndSales } from '../../features/trading/components/TimeAndSales';

const TradeExecutionPage: React.FC = () => {
  return (
    <Box sx={{ height: 'calc(100vh - 120px)', display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Box sx={{ flexGrow: 1, display: 'flex', gap: 2, minHeight: 0 }}>
        {/* Depth of Market - Left Column */}
        <Paper sx={{ 
          flex: 1.2, 
          display: 'flex', 
          flexDirection: 'column',
          bgcolor: 'rgba(10, 11, 20, 0.4)',
          border: `1px solid ${alpha('#00ffa3', 0.2)}`,
          borderRadius: 1,
          overflow: 'hidden'
        }}>
          <DepthOfMarket />
        </Paper>

        {/* Level II Quotes - Middle Column */}
        <Paper sx={{ 
          flex: 1, 
          display: 'flex', 
          flexDirection: 'column',
          bgcolor: 'rgba(10, 11, 20, 0.4)',
          border: `1px solid ${alpha('#ff2e7e', 0.2)}`,
          borderRadius: 1,
          overflow: 'hidden'
        }}>
          <LevelIIQuotes />
        </Paper>

        {/* Order Ticket - Right Column */}
        <Paper sx={{ 
          flex: 1, 
          display: 'flex', 
          flexDirection: 'column',
          bgcolor: 'rgba(10, 11, 20, 0.4)',
          border: `1px solid ${alpha('#3bc1ff', 0.1)}`,
          borderRadius: 1,
          overflow: 'hidden'
        }}>
          <OrderTicket />
        </Paper>
      </Box>

      {/* Time & Sales - Bottom Bar */}
      <TimeAndSales />
    </Box>
  );
};

export default TradeExecutionPage;
