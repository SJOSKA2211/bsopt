import React, { useState } from 'react';
import {
  Box,
  CssBaseline,
  Typography,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Paper,
  alpha,
  useTheme,
  Stack,
} from '@mui/material';
import {
  DashboardOutlined as DashboardIcon,
  SwapHorizOutlined as TransfersIcon,
  PaymentsOutlined as PaymentsIcon,
  SportsEsportsOutlined as GamesIcon,
  ConfirmationNumberOutlined as TicketsIcon,
  AccountBalanceWalletOutlined as WalletIcon,
  MailOutline as MessagesIcon,
  NotificationsNoneOutlined as NotificationsIcon,
  SettingsOutlined as SettingsIcon,
  ExitToAppOutlined as LogoutIcon,
  Wallet as BrandIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';

const drawerWidth = 260;

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const theme = useTheme();
  const navigate = useNavigate();
  const location = useLocation();

  const navigationItems = [
    { type: 'header', text: 'GENERAL' },
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
    { text: 'Transfers', icon: <TransfersIcon />, path: '/transfers' },
    { text: 'Payments', icon: <PaymentsIcon />, path: '/payments' },
    { text: 'Games', icon: <GamesIcon />, path: '/games' },
    { text: 'Tickets', icon: <TicketsIcon />, path: '/tickets' },
    { type: 'header', text: 'PERSONAL', spacing: 3 },
    { text: 'Wallet', icon: <WalletIcon />, path: '/portfolio' },
    { text: 'Messages', icon: <MessagesIcon />, path: '/messages' },
    { text: 'Notifications', icon: <NotificationsIcon />, path: '/notifications' },
    { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
  ];

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
      <CssBaseline />
      
      {/* Sidebar Navigation */}
      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
            borderRight: `1px solid ${alpha(theme.palette.divider, 0.05)}`,
            display: 'flex',
            flexDirection: 'column',
            px: 2,
            py: 3,
            bgcolor: 'background.default',
          },
        }}
      >
        {/* Brand */}
        <Stack 
          direction="row" 
          spacing={1.5} 
          alignItems="center" 
          sx={{ px: 2, mb: 4, cursor: 'pointer' }}
          onClick={() => navigate('/')}
        >
          <Box
            sx={{
              width: 32,
              height: 32,
              borderRadius: 1,
              bgcolor: 'text.primary',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <BrandIcon sx={{ color: 'background.default', fontSize: 20 }} />
          </Box>
          <Typography variant="h6" sx={{ fontWeight: 800, letterSpacing: '0.05em', fontSize: '1.1rem' }}>
            CASHMATE
          </Typography>
        </Stack>

        <Box sx={{ overflowY: 'auto', flexGrow: 1, '&::-webkit-scrollbar': { display: 'none' } }}>
          <List disablePadding>
            {navigationItems.map((item, index) => {
              if (item.type === 'header') {
                return (
                  <Typography
                    key={item.text}
                    variant="caption"
                    sx={{
                      display: 'block',
                      px: 2,
                      mt: item.spacing || 0,
                      mb: 1,
                      fontWeight: 700,
                      color: 'text.disabled',
                      letterSpacing: '0.1em',
                    }}
                  >
                    {item.text}
                  </Typography>
                );
              }
              const isActive = location.pathname === item.path;
              return (
                <ListItem key={item.text} disablePadding sx={{ mb: 0.5 }}>
                  <ListItemButton
                    selected={isActive}
                    onClick={() => navigate(item.path || '/')}
                    sx={{
                      borderRadius: 2,
                      py: 1.25,
                      '&.Mui-selected': {
                        bgcolor: alpha(theme.palette.primary.main, 0.1),
                        color: 'primary.main',
                        '& .MuiListItemIcon-root': { color: 'primary.main' },
                        '&:hover': { bgcolor: alpha(theme.palette.primary.main, 0.15) },
                      },
                    }}
                  >
                    <ListItemIcon sx={{ minWidth: 40, color: isActive ? 'primary.main' : 'text.secondary' }}>
                      {item.icon}
                    </ListItemIcon>
                    <ListItemText 
                      primary={item.text} 
                      primaryTypographyProps={{ 
                        fontSize: '0.9rem', 
                        fontWeight: isActive ? 600 : 500 
                      }} 
                    />
                  </ListItemButton>
                </ListItem>
              );
            })}
          </List>
        </Box>

        {/* Footer Sidebar Widgets */}
        <Stack spacing={2} sx={{ mt: 'auto', pt: 2 }}>
          <Paper
            sx={{
              p: 2,
              borderRadius: 3,
              bgcolor: alpha(theme.palette.background.paper, 0.4),
              border: `1px solid ${alpha(theme.palette.divider, 0.05)}`,
              boxShadow: 'none',
            }}
          >
            <Typography variant="caption" sx={{ color: 'text.disabled', fontWeight: 600 }}>
              MONTHLY CASHBACK
            </Typography>
            <Typography variant="h6" sx={{ fontWeight: 700, mt: 0.5 }}>
              $215.50
            </Typography>
          </Paper>

          <ListItemButton sx={{ borderRadius: 2, py: 1.5 }}>
            <ListItemIcon sx={{ minWidth: 40 }}>
              <LogoutIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Log out" primaryTypographyProps={{ fontSize: '0.9rem', fontWeight: 500 }} />
          </ListItemButton>
        </Stack>
      </Drawer>

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          height: '100vh',
          overflow: 'auto',
          p: { xs: 2, md: 4 },
          background: theme.palette.background.default,
        }}
      >
        {children}
      </Box>
    </Box>
  );
};
