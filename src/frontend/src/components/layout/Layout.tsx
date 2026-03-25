import React from 'react';
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
} from '@mui/material';
import {
  GridViewOutlined as DashboardIcon,
  BarChartOutlined as TradeIcon,
  AccountBalanceWalletOutlined as PositionsIcon,
  TimelineOutlined as AnalysisIcon,
  HistoryOutlined as HistoryIcon,
  SettingsOutlined as SettingsIcon,
  NotificationsNoneOutlined as NotifIcon,
  AccountCircleOutlined as UserIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { stitchTokens } from '../../theme/stitch-tokens';

const drawerWidth = 260;

interface NavItemProps {
  item: { text: string; icon: React.ReactNode; path: string };
  isActive: boolean;
  onClick: () => void;
}

const NavItem: React.FC<NavItemProps> = ({ item, isActive, onClick }) => {
  return (
    <ListItem disablePadding sx={{ mb: 0.5, px: 2 }}>
      <ListItemButton
        selected={isActive}
        onClick={onClick}
        sx={{
          borderRadius: 0,
          py: 1.2,
          position: 'relative',
          transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
          backgroundColor: isActive ? alpha(stitchTokens.colors.primary, 0.1) : 'transparent',
          borderLeft: isActive ? `3px solid ${stitchTokens.colors.primary}` : '3px solid transparent',
          '&:hover': {
            backgroundColor: alpha(stitchTokens.colors.primary, 0.05),
            borderLeft: isActive ? `3px solid ${stitchTokens.colors.primary}` : `3px solid ${alpha(stitchTokens.colors.primary, 0.3)}`,
          },
          '&.Mui-selected': {
            backgroundColor: alpha(stitchTokens.colors.primary, 0.1),
          },
        }}
      >
        <ListItemIcon
          sx={{
            minWidth: 36,
            color: isActive ? stitchTokens.colors.primary : '#a9abb1',
          }}
        >
          <Box sx={{ fontSize: 20, display: 'flex' }}>
            {item.icon}
          </Box>
        </ListItemIcon>
        <ListItemText
          primary={item.text}
          primaryTypographyProps={{
            fontSize: '0.85rem',
            fontWeight: isActive ? 700 : 500,
            color: isActive ? stitchTokens.colors.primary : '#f5f6fc',
            fontFamily: stitchTokens.typography.labels,
            textTransform: 'uppercase',
            letterSpacing: '0.1em',
          }}
        />
      </ListItemButton>
    </ListItem>
  );
};

export const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const navItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
    { text: 'Trade', icon: <TradeIcon />, path: '/market' },
    { text: 'Optimizer', icon: <AnalysisIcon />, path: '/research' },
    { text: 'Positions', icon: <PositionsIcon />, path: '/portfolio' },
    { text: 'Risk', icon: <AnalysisIcon />, path: '/risk' },
    { text: 'History', icon: <HistoryIcon />, path: '/history' },
    { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
  ];

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: stitchTokens.colors.background }}>
      <CssBaseline />

      {/* Sidebar - The Terminal Core */}
      <Drawer
        variant="permanent"
        PaperProps={{
          sx: {
            width: drawerWidth,
            bgcolor: 'rgba(16, 20, 24, 0.95)',
            backdropFilter: stitchTokens.effects.glassBlur,
            borderRight: stitchTokens.effects.glassBorder,
            backgroundImage: 'none',
            display: 'flex',
            flexDirection: 'column',
          }
        }}
        sx={{ width: drawerWidth, flexShrink: 0 }}
      >
        {/* Brand Shard */}
        <Box sx={{ p: 0, mt: 4, mb: 4 }}>
          <Box className="stitch-slanted-header" sx={{ width: 'fit-content' }}>
            BS-OPT V2.4
          </Box>
        </Box>

        {/* Navigation List */}
        <Box sx={{ flexGrow: 1 }}>
          <List disablePadding>
            {navItems.map((item) => (
              <NavItem
                key={item.text}
                item={item}
                isActive={location.pathname === item.path}
                onClick={() => navigate(item.path)}
              />
            ))}
          </List>
        </Box>

        {/* System Health / User Block */}
        <Box sx={{ p: 2, borderTop: stitchTokens.effects.glassBorder }}>
          <Stack spacing={2}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
              <div className="stitch-live-indicator" />
              <Typography className="stitch-label" sx={{ fontSize: '9px' }}>
                System Ready // Latency: 4ms
              </Typography>
            </Box>
            
            <Stack direction="row" spacing={1.5} alignItems="center" sx={{ p: 1.5, className: "stitch-card" }}>
              <Avatar 
                sx={{ 
                  width: 32, 
                  height: 32, 
                  bgcolor: alpha(stitchTokens.colors.primary, 0.1),
                  border: `1px solid ${alpha(stitchTokens.colors.primary, 0.3)}`,
                  fontSize: '0.8rem',
                  fontWeight: 900,
                  color: stitchTokens.colors.primary
                }}
              >
                QT
              </Avatar>
              <Box>
                <Typography sx={{ fontSize: '0.75rem', fontWeight: 700, lineHeight: 1 }}>
                  Quant Trader
                </Typography>
                <Typography className="stitch-label" sx={{ fontSize: '8px', mt: 0.5 }}>
                  Institutional Tier
                </Typography>
              </Box>
              <IconButton size="small" sx={{ ml: 'auto', color: '#a9abb1' }}>
                <SettingsIcon sx={{ fontSize: 16 }} />
              </IconButton>
            </Stack>
          </Stack>
        </Box>
      </Drawer>

      {/* Main Content Area */}
      <Box component="main" sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Market Pulse Top Bar */}
        <Box sx={{ 
          height: 72, 
          px: 3, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between',
          borderBottom: stitchTokens.effects.glassBorder,
          bgcolor: 'rgba(0,0,0,0.4)',
          backdropFilter: 'blur(10px)',
          zIndex: 1100
        }}>
          <Stack direction="row" spacing={4} alignItems="center">
            <Box className="stitch-card" sx={{ p: '6px 16px', borderLeft: `2px solid ${stitchTokens.colors.primary}` }}>
              <Typography className="stitch-label" sx={{ fontSize: '9px', mb: 0.2 }}>Active Symbol</Typography>
              <Typography sx={{ fontWeight: 900, fontSize: '1rem', letterSpacing: '0.05em' }}>
                AAPL <Typography component="span" variant="caption" sx={{ color: '#a9abb1', ml: 1 }}>$189.45 / -0.2%</Typography>
              </Typography>
            </Box>

            <Stack direction="row" spacing={3}>
              {[
                { label: 'MARKET_IV', value: '28.4%', color: stitchTokens.colors.primary },
                { label: 'CAL_SPREAD', value: '1.45', color: '#f5f6fc' },
                { label: 'GAMMA_EXP', value: '+1.2M', color: stitchTokens.colors.tertiary },
              ].map(ticker => (
                <Box key={ticker.label}>
                  <Typography className="stitch-label" sx={{ fontSize: '8px', opacity: 0.7 }}>{ticker.label}</Typography>
                  <Typography className="stitch-mono" sx={{ fontSize: '12px', fontWeight: 800, color: ticker.color }}>
                    {ticker.value}
                  </Typography>
                </Box>
              ))}
            </Stack>
          </Stack>

          <Stack direction="row" spacing={1} alignItems="center">
             <Box sx={{ textAlign: 'right', mr: 2 }}>
                <Typography className="stitch-label" sx={{ fontSize: '8px' }}>Global Portfolio</Typography>
                <Typography className="stitch-mono" sx={{ fontSize: '14px', fontWeight: 900, color: stitchTokens.colors.primary }}>
                  $2,450,192.40
                </Typography>
             </Box>
            <IconButton sx={{ color: '#f5f6fc', animation: 'none' }}>
              <NotifIcon />
            </IconButton>
          </Stack>
        </Box>

        {/* Content Wrapper */}
        <Box sx={{ flexGrow: 1, p: 0, overflow: 'auto', position: 'relative' }}>
          {children}
        </Box>
      </Box>
    </Box>
  );
};
