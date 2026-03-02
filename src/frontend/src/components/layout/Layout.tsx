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
  useTheme,
  Divider,
  Avatar,
  Chip,
} from '@mui/material';
import {
  DashboardOutlined as DashboardIcon,
  ShowChartOutlined as MarketIcon,
  AccountBalanceWalletOutlined as PortfolioIcon,
  SettingsOutlined as SettingsIcon,
  ExitToAppOutlined as LogoutIcon,
  TrendingUpOutlined as BrandIcon,
  NotificationsNoneOutlined as NotifIcon,
  PersonOutlined as PersonIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';

const drawerWidth = 264;

// ---------------------------------------------------------------------------
// Live market ticker data
// ---------------------------------------------------------------------------
const TICKERS = [
  { symbol: 'AAPL', price: '189.42', change: '+2.18', pct: '+1.17%', up: true },
  { symbol: 'SPY', price: '471.23', change: '-1.42', pct: '-0.30%', up: false },
  { symbol: 'QQQ', price: '401.15', change: '+6.72', pct: '+1.70%', up: true },
  { symbol: 'NVDA', price: '492.80', change: '+15.6', pct: '+3.25%', up: true },
  { symbol: 'TSLA', price: '248.12', change: '-4.33', pct: '-1.72%', up: false },
  { symbol: 'SPX', price: '5127.3', change: '+18.4', pct: '+0.36%', up: true },
];

// ---------------------------------------------------------------------------
// Ticker strip component
// ---------------------------------------------------------------------------
const TickerStrip: React.FC = () => {
  const theme = useTheme();
  const doubled = [...TICKERS, ...TICKERS]; // duplicate for seamless loop

  return (
    <Box
      sx={{
        width: '100%',
        height: 40,
        bgcolor: alpha(theme.palette.background.paper, 0.7),
        backdropFilter: 'blur(12px)',
        borderBottom: `1px solid ${alpha('#94a3b8', 0.08)}`,
        overflow: 'hidden',
        display: 'flex',
        alignItems: 'center',
        flexShrink: 0,
      }}
    >
      <Box className="ticker-strip">
        <Box className="ticker-track">
          {doubled.map((t, i) => (
            <Box key={i} className="ticker-item">
              <Typography
                variant="caption"
                sx={{
                  fontWeight: 700,
                  color: 'text.primary',
                  fontFamily: '"JetBrains Mono", monospace',
                  fontSize: '0.7rem',
                  letterSpacing: '0.05em',
                }}
              >
                {t.symbol}
              </Typography>
              <Typography
                variant="caption"
                sx={{
                  fontFamily: '"JetBrains Mono", monospace',
                  fontSize: '0.7rem',
                  color: 'text.secondary',
                }}
              >
                {t.price}
              </Typography>
              <Typography
                variant="caption"
                sx={{
                  fontFamily: '"JetBrains Mono", monospace',
                  fontSize: '0.65rem',
                  color: t.up ? 'success.main' : 'error.main',
                  fontWeight: 600,
                }}
              >
                {t.pct}
              </Typography>
            </Box>
          ))}
        </Box>
      </Box>
    </Box>
  );
};

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const theme = useTheme();
  const navigate = useNavigate();
  const location = useLocation();

  const isLogin = location.pathname === '/login';

  const navSections = [
    {
      header: 'TERMINAL',
      items: [
        { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
        { text: 'Market', icon: <MarketIcon />, path: '/market' },
        { text: 'Portfolio', icon: <PortfolioIcon />, path: '/portfolio' },
      ],
    },
    {
      header: 'ACCOUNT',
      items: [
        { text: 'Notifications', icon: <NotifIcon />, path: '/notifications' },
        { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
      ],
    },
  ];

  // No sidebar on login
  if (isLogin) {
    return (
      <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
        {children}
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
      <CssBaseline />

      {/* ------------------------------------------------------------------ */}
      {/* Sidebar                                                             */}
      {/* ------------------------------------------------------------------ */}
      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
            display: 'flex',
            flexDirection: 'column',
            px: 1.5,
            py: 3,
          },
        }}
      >
        {/* Brand */}
        <Stack
          direction="row"
          spacing={1.5}
          alignItems="center"
          sx={{
            px: 1.5,
            mb: 4,
            cursor: 'pointer',
            '&:hover': { opacity: 0.85 },
            transition: 'opacity 0.2s ease',
          }}
          onClick={() => navigate('/')}
        >
          <Box
            sx={{
              width: 36,
              height: 36,
              borderRadius: 2,
              background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: `0 4px 14px ${alpha('#10b981', 0.4)}`,
            }}
          >
            <BrandIcon sx={{ color: '#fff', fontSize: 20 }} />
          </Box>
          <Box>
            <Typography
              variant="h6"
              sx={{
                fontWeight: 800,
                letterSpacing: '-0.01em',
                fontSize: '1.05rem',
                background: 'linear-gradient(135deg, #10b981, #38bdf8)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
              }}
            >
              BS-Opt
            </Typography>
            <Typography
              variant="caption"
              sx={{
                color: 'text.disabled',
                fontSize: '0.62rem',
                letterSpacing: '0.08em',
                display: 'block',
                mt: -0.25,
              }}
            >
              OPTIONS TERMINAL
            </Typography>
          </Box>
        </Stack>

        {/* Market status badge */}
        <Box sx={{ px: 1.5, mb: 3 }}>
          <Chip
            icon={
              <Box
                component="span"
                sx={{
                  width: 6,
                  height: 6,
                  borderRadius: '50%',
                  bgcolor: 'success.main',
                  boxShadow: `0 0 6px ${alpha('#10b981', 0.8)}`,
                  animation: 'live-pulse 1.8s ease-in-out infinite',
                  '@keyframes live-pulse': {
                    '0%, 100%': { opacity: 1 },
                    '50%': { opacity: 0.4 },
                  },
                  flexShrink: 0,
                }}
              />
            }
            label="MARKET OPEN"
            size="small"
            sx={{
              bgcolor: alpha('#10b981', 0.1),
              color: 'success.main',
              border: `1px solid ${alpha('#10b981', 0.2)}`,
              fontWeight: 700,
              fontSize: '0.62rem',
              letterSpacing: '0.07em',
              height: 24,
              '& .MuiChip-icon': { ml: 1, mr: -0.5 },
            }}
          />
        </Box>

        {/* Navigation */}
        <Box sx={{ flexGrow: 1, overflowY: 'auto', '&::-webkit-scrollbar': { display: 'none' } }}>
          {navSections.map((section) => (
            <Box key={section.header} sx={{ mb: 3 }}>
              <Typography
                variant="caption"
                sx={{
                  display: 'block',
                  px: 2,
                  mb: 0.75,
                  fontWeight: 700,
                  color: 'text.disabled',
                  letterSpacing: '0.12em',
                  fontSize: '0.62rem',
                }}
              >
                {section.header}
              </Typography>
              <List disablePadding>
                {section.items.map((item) => {
                  const isActive = location.pathname === item.path;
                  return (
                    <ListItem key={item.text} disablePadding sx={{ mb: 0.25 }}>
                      <ListItemButton
                        selected={isActive}
                        onClick={() => navigate(item.path)}
                        sx={{
                          borderRadius: 2,
                          py: 1.1,
                          position: 'relative',
                          overflow: 'hidden',
                          ...(isActive && {
                            '&::before': {
                              content: '""',
                              position: 'absolute',
                              left: 0,
                              top: '20%',
                              bottom: '20%',
                              width: 3,
                              borderRadius: '0 3px 3px 0',
                              background: 'linear-gradient(180deg, #10b981, #38bdf8)',
                              boxShadow: `0 0 8px ${alpha('#10b981', 0.6)}`,
                            },
                          }),
                        }}
                      >
                        <ListItemIcon
                          sx={{
                            minWidth: 36,
                            color: isActive ? 'primary.main' : 'text.disabled',
                            transition: 'color 0.18s ease',
                          }}
                        >
                          {item.icon}
                        </ListItemIcon>
                        <ListItemText
                          primary={item.text}
                          primaryTypographyProps={{
                            fontSize: '0.875rem',
                            fontWeight: isActive ? 700 : 500,
                            color: isActive ? 'primary.main' : 'text.secondary',
                          }}
                        />
                      </ListItemButton>
                    </ListItem>
                  );
                })}
              </List>
            </Box>
          ))}
        </Box>

        {/* Footer */}
        <Divider sx={{ mb: 2 }} />
        <Stack spacing={0.5}>
          {/* User info */}
          <Stack
            direction="row"
            spacing={1.5}
            alignItems="center"
            sx={{
              px: 1.5,
              py: 1.25,
              borderRadius: 2,
              cursor: 'pointer',
              transition: 'background 0.18s ease',
              '&:hover': { bgcolor: alpha('#94a3b8', 0.07) },
            }}
          >
            <Avatar
              sx={{
                width: 34,
                height: 34,
                background: 'linear-gradient(135deg, #a855f7, #38bdf8)',
                fontSize: '0.8rem',
                fontWeight: 700,
              }}
            >
              <PersonIcon sx={{ fontSize: 18 }} />
            </Avatar>
            <Box sx={{ flexGrow: 1, overflow: 'hidden' }}>
              <Typography variant="body2" sx={{ fontWeight: 600, fontSize: '0.8rem' }}>
                Quant Trader
              </Typography>
              <Typography variant="caption" sx={{ color: 'text.disabled', fontSize: '0.67rem' }} noWrap>
                trader@bsopt.io
              </Typography>
            </Box>
          </Stack>

          <ListItemButton sx={{ borderRadius: 2 }} onClick={() => navigate('/login')}>
            <ListItemIcon sx={{ minWidth: 36, color: 'text.disabled' }}>
              <LogoutIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText
              primary="Sign out"
              primaryTypographyProps={{ fontSize: '0.875rem', fontWeight: 500, color: 'text.secondary' }}
            />
          </ListItemButton>
        </Stack>
      </Drawer>

      {/* ------------------------------------------------------------------ */}
      {/* Main content area                                                   */}
      {/* ------------------------------------------------------------------ */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          height: '100vh',
          overflow: 'auto',
          display: 'flex',
          flexDirection: 'column',
          bgcolor: 'background.default',
        }}
      >
        {/* Live ticker at the very top */}
        <TickerStrip />

        <Box sx={{ flexGrow: 1, p: { xs: 2, md: 4 }, overflow: 'auto' }}>
          {children}
        </Box>
      </Box>
    </Box>
  );
};
