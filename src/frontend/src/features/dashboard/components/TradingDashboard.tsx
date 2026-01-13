import React, { useState } from 'react';
import {
  Box,
  CssBaseline,
  AppBar,
  Toolbar,
  Typography,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  IconButton,
  Grid,
  Paper,
  Container,
  alpha,
  useTheme,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  ShowChart as ChartIcon,
  AccountBalanceWallet as PortfolioIcon,
  Settings as SettingsIcon,
  AccountCircle,
} from '@mui/icons-material';
import { OptionsChain } from '../../options/components/OptionsChain';
import { PortfolioSummary } from '../../portfolio/components/PortfolioSummary';
import { LivePriceChart } from '../../charts/components/LivePriceChart';

const drawerWidth = 240;

export const TradingDashboard: React.FC = () => {
  const theme = useTheme();
  const [drawerOpen, setDrawerOpen] = useState(true);

  const toggleDrawer = () => {
    setDrawerOpen(!drawerOpen);
  };

  return (
    <Box sx={{ display: 'flex', height: '100vh', bgcolor: 'background.default' }}>
      <CssBaseline />
      
      {/* Header */}
      <AppBar
        position="fixed"
        sx={{
          zIndex: theme.zIndex.drawer + 1,
          bgcolor: alpha(theme.palette.background.paper, 0.8),
          backdropFilter: 'blur(8px)',
          borderBottom: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
          boxShadow: 'none',
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={toggleDrawer}
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1, fontWeight: 'bold', color: 'primary.main' }}>
            BS-Opt Trading Dashboard
          </Typography>
          <IconButton color="inherit">
            <AccountCircle />
          </IconButton>
        </Toolbar>
      </AppBar>

      {/* Sidebar Navigation */}
      <Drawer
        variant="permanent"
        open={drawerOpen}
        sx={{
          width: drawerOpen ? drawerWidth : 72,
          flexShrink: 0,
          whiteSpace: 'nowrap',
          boxSizing: 'border-box',
          '& .MuiDrawer-paper': {
            width: drawerOpen ? drawerWidth : 72,
            overflowX: 'hidden',
            transition: theme.transitions.create('width', {
              easing: theme.transitions.easing.sharp,
              duration: theme.transitions.duration.enteringScreen,
            }),
            bgcolor: 'background.paper',
            borderRight: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
          },
        }}
        role="navigation"
      >
        <Toolbar />
        <Box sx={{ overflow: 'auto' }}>
          <List>
            {[
              { text: 'Dashboard', icon: <DashboardIcon /> },
              { text: 'Market Data', icon: <ChartIcon /> },
              { text: 'Portfolio', icon: <PortfolioIcon /> },
              { text: 'Settings', icon: <SettingsIcon /> },
            ].map((item) => (
              <ListItem button key={item.text}>
                <ListItemIcon sx={{ color: 'primary.main', minWidth: 0, mr: drawerOpen ? 3 : 'auto', justifyContent: 'center' }}>
                  {item.icon}
                </ListItemIcon>
                {drawerOpen && <ListItemText primary={item.text} />}
              </ListItem>
            ))}
          </List>
        </Box>
      </Drawer>

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          height: '100vh',
          overflow: 'auto',
          pt: 8,
          pb: 2,
          px: 2,
        }}
      >
        <Container maxWidth="xl" sx={{ mt: 2, mb: 2 }}>
          <Grid container spacing={3}>
            {/* Real-Time Price Chart */}
            <Grid item xs={12} lg={12}>
              <Paper
                data-testid="live-price-chart-paper"
                sx={{
                  p: 2,
                  display: 'flex',
                  flexDirection: 'column',
                  height: 450,
                }}
              >
                <Typography variant="h6" gutterBottom>
                  Real-Time Price Chart - AAPL
                </Typography>
                <Box sx={{ flex: 1, overflow: 'hidden' }}>
                  <LivePriceChart symbol="AAPL" />
                </Box>
              </Paper>
            </Grid>

            {/* Options Chain Section */}
            <Grid item xs={12} lg={8}>
              <Paper
                data-testid="options-chain-container"
                sx={{
                  p: 0,
                  display: 'flex',
                  flexDirection: 'column',
                  height: 600,
                  overflow: 'hidden',
                }}
              >
                <OptionsChain symbol="AAPL" />
              </Paper>
            </Grid>

            {/* Portfolio Summary Section */}
            <Grid item xs={12} lg={4}>
              <Paper
                data-testid="portfolio-summary-container"
                sx={{
                  p: 0,
                  display: 'flex',
                  flexDirection: 'column',
                  height: 600,
                  overflow: 'hidden',
                }}
              >
                <PortfolioSummary />
              </Paper>
            </Grid>
          </Grid>
        </Container>
      </Box>
    </Box>
  );
};
