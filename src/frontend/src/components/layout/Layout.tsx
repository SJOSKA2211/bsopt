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
  Stack, 
  alpha, 
  Avatar, 
  IconButton, 
  useMediaQuery,
  useTheme
} from '@mui/material';
import {
  GridViewOutlined as DashboardIcon,
  BarChartOutlined as TradeIcon,
  AccountBalanceWalletOutlined as PositionsIcon,
  TimelineOutlined as AnalysisIcon,
  HistoryOutlined as HistoryIcon,
  SettingsOutlined as SettingsIcon,
  NotificationsNoneOutlined as NotifIcon,
  Bolt as FlashIcon,
  LogoutOutlined as LogoutIcon,
  Menu as MenuIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { TickerTape } from '../TickerTape';
import { useGatewayHealth } from '../../hooks/useGatewayHealth';
import { AnimatedCard } from '../common/AnimatedCard';

const drawerWidth = 280;

interface NavItemProps {
  item: { text: string; icon: React.ReactNode; path: string; color?: string };
  isActive: boolean;
  onClick: () => void;
  index: number;
}

const NavItem: React.FC<NavItemProps> = ({ item, isActive, onClick, index }) => {
  return (
    <motion.div
      initial={{ x: -20, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ delay: 0.1 * index }}
    >
      <ListItem disablePadding sx={{ mb: 1, px: 2 }}>
        <ListItemButton
          selected={isActive}
          onClick={onClick}
          sx={{
            borderRadius: '12px',
            py: 1.5,
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            backgroundColor: isActive ? 'rgba(255, 255, 255, 0.05)' : 'transparent',
            '&:hover': {
              backgroundColor: 'rgba(255, 255, 255, 0.08)',
              transform: 'translateX(4px)',
            },
          }}
        >
          <ListItemIcon
            sx={{
              minWidth: 40,
              color: isActive ? 'var(--accent-mint)' : 'var(--text-secondary)',
            }}
          >
            <Box sx={{ fontSize: 22, display: 'flex' }}>
              {item.icon}
            </Box>
          </ListItemIcon>
          <ListItemText
            primary={item.text}
            primaryTypographyProps={{
              fontSize: '13px',
              fontWeight: isActive ? 700 : 500,
              color: isActive ? '#fff' : 'var(--text-secondary)',
              letterSpacing: '0.02em',
            }}
          />
          {isActive && (
            <motion.div
              layoutId="nav-glow"
              style={{
                position: 'absolute',
                right: 0,
                width: 4,
                height: 20,
                background: 'var(--accent-mint)',
                borderRadius: '4px 0 0 4px',
                boxShadow: '0 0 10px var(--accent-mint)',
              }}
            />
          )}
        </ListItemButton>
      </ListItem>
    </motion.div>
  );
};

export const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [mobileOpen, setMobileOpen] = useState(false);
  const health = useGatewayHealth();

  const navItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
    { text: 'Market', icon: <TradeIcon />, path: '/market' },
    { text: 'Optimizer', icon: <AnalysisIcon />, path: '/research' },
    { text: 'Portfolio', icon: <PositionsIcon />, path: '/portfolio' },
    { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
  ];

  const SidebarContent = () => (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', p: '24px' }}>
      {/* Brand */}
      <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 6, px: 1 }}>
        <FlashIcon sx={{ color: 'var(--accent-mint)', fontSize: 24 }} />
        <Typography variant="h6" sx={{ fontWeight: 800, letterSpacing: '-0.02em' }}>
          BSOPT_V2
        </Typography>
      </Stack>

      {/* Navigation */}
      <Box sx={{ flexGrow: 1 }}>
        <List disablePadding>
          {navItems.map((item, index) => (
            <NavItem
              key={item.text}
              item={item}
              isActive={location.pathname === item.path}
              onClick={() => {
                navigate(item.path);
                if (isMobile) setMobileOpen(false);
              }}
              index={index}
            />
          ))}
        </List>
      </Box>

      {/* User & Health */}
      <AnimatedCard sx={{ p: 2, background: 'rgba(255,255,255,0.03)' }}>
        <Stack spacing={2}>
          <Stack direction="row" spacing={1.5} alignItems="center">
            <Avatar sx={{ width: 32, height: 32, bgcolor: alpha('#fff', 0.1), fontSize: '12px', fontWeight: 700 }}>QT</Avatar>
            <Box>
              <Typography sx={{ fontSize: '12px', fontWeight: 700 }}>Trader_Alpha</Typography>
              <Typography variant="caption" sx={{ color: 'var(--text-secondary)' }}>Inst_Access</Typography>
            </Box>
          </Stack>
          <Box sx={{ p: 1.5, background: 'rgba(0,0,0,0.2)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <Stack direction="row" spacing={1} alignItems="center">
              <Box className="status-pill healthy" sx={{ width: 8, height: 8, p: 0, borderRadius: '50%', background: 'var(--accent-mint)' }} />
              <Typography sx={{ fontSize: '10px', fontWeight: 600, color: 'var(--accent-mint)' }}>
                RT_FEED: {health.latency}ms
              </Typography>
            </Stack>
          </Box>
        </Stack>
      </AnimatedCard>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', background: 'var(--bento-bg)' }}>
      <CssBaseline />

      {!isMobile && (
        <Box sx={{ width: drawerWidth, flexShrink: 0, borderRight: '1px solid var(--bento-card-border)' }}>
          <SidebarContent />
        </Box>
      )}

      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={() => setMobileOpen(false)}
        sx={{
          '& .MuiDrawer-paper': { width: drawerWidth, background: 'var(--bento-bg)' },
        }}
      >
        <SidebarContent />
      </Drawer>

      <Box component="main" sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', position: 'relative' }}>
        {/* Header */}
        <Box sx={{ 
          height: 72, 
          px: 3, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between',
          borderBottom: '1px solid var(--bento-card-border)',
          background: 'rgba(5, 5, 5, 0.5)',
          backdropFilter: 'blur(10px)',
          zIndex: 10
        }}>
          <Stack direction="row" spacing={2} alignItems="center">
            {isMobile && (
              <IconButton onClick={() => setMobileOpen(true)} sx={{ color: '#fff' }}>
                <MenuIcon />
              </IconButton>
            )}
            <Box className="status-pill" sx={{ background: 'rgba(255,255,255,0.05)', border: '1px solid var(--bento-card-border)' }}>
               <Typography sx={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-secondary)' }}>
                 NODE: <Box component="span" sx={{ color: '#fff' }}>QUANT_042</Box>
               </Typography>
            </Box>
          </Stack>

          <Stack direction="row" spacing={2} alignItems="center">
             <IconButton sx={{ color: 'var(--text-secondary)' }}>
                <NotifIcon />
             </IconButton>
             <IconButton sx={{ color: 'var(--text-secondary)' }}>
                <LogoutIcon />
             </IconButton>
          </Stack>
        </Box>

        <TickerTape />

        <Box sx={{ flexGrow: 1, overflow: 'hidden' }}>
           <AnimatePresence mode="wait">
              <motion.div
                key={location.pathname}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.3 }}
                style={{ height: '100%' }}
              >
                 {children}
              </motion.div>
           </AnimatePresence>
        </Box>
      </Box>
    </Box>
  );
};
