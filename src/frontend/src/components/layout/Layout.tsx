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
import { useMarketTickers } from '../../api/hooks';
import type { Ticker } from '../../api/types';

// ---------------------------------------------------------------------------
// Ticker strip component
// ---------------------------------------------------------------------------
const TickerStrip: React.FC = () => {
  const theme = useTheme();
  const financial = theme.palette.financial;
  const qfd = financial?.qfd;
  
  const { data: tickers, isLoading } = useMarketTickers();

  // Seamless institutional loop - duplicate for CSS animation
  const displayTickers = tickers ? [...tickers, ...tickers] : [];

  if (isLoading && !tickers) {
    return (
      <Box sx={{ width: '100%', height: 40, bgcolor: alpha(theme.palette.background.paper, 0.4), backdropFilter: 'blur(20px)', borderBottom: `1px solid ${alpha('#fff', 0.05)}`, display: 'flex', alignItems: 'center', px: 3 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 800, letterSpacing: '0.1em' }}>SYNCHRONIZING GLOBAL TAPE...</Typography>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        width: '100%',
        height: 40,
        bgcolor: alpha(theme.palette.background.paper, 0.4),
        backdropFilter: 'blur(30px)',
        borderBottom: `1px solid ${alpha('#fff', 0.03)}`,
        overflow: 'hidden',
        display: 'flex',
        alignItems: 'center',
        flexShrink: 0,
      }}
    >
      <Box className="ticker-strip">
        <Box className="ticker-track">
          {displayTickers.map((t: Ticker, i: number) => (
            <Box key={`${t.symbol}-${i}`} className="ticker-item">
              <Typography
                variant="caption"
                sx={{
                  fontWeight: 900,
                  color: 'text.primary',
                  fontFamily: 'Outfit',
                  fontSize: '0.75rem',
                  letterSpacing: '0.02em',
                }}
              >
                {t.symbol}
              </Typography>
              <Typography
                variant="caption"
                sx={{
                  fontFamily: 'JetBrains Mono',
                  fontSize: '0.7rem',
                  color: 'text.secondary',
                  fontWeight: 600,
                }}
              >
                ${parseFloat(t.price).toFixed(2)}
              </Typography>
              <Typography
                variant="caption"
                sx={{
                  fontFamily: 'JetBrains Mono',
                  fontSize: '0.65rem',
                  color: t.up ? qfd?.emerald ?? '#10b981' : theme.palette.error.main,
                  fontWeight: 900,
                }}
              >
                {t.percentChange}
              </Typography>
            </Box>
          ))}
        </Box>
      </Box>
    </Box>
  );
};

interface NavItemProps {
  item: { text: string; icon: React.ReactNode; path: string };
  isActive: boolean;
  onClick: () => void;
}

const NavItem: React.FC<NavItemProps> = ({ item, isActive, onClick }) => {
  const theme = useTheme();
  return (
    <ListItem disablePadding sx={{ mb: 0.25 }}>
      <ListItemButton
        selected={isActive}
        onClick={onClick}
        sx={{
          borderRadius: 2,
          py: 1.1,
          position: 'relative',
          overflow: 'hidden',
          ...(isActive && {
            bgcolor: alpha(theme.palette.primary.main, 0.08),
            '&::before': {
              content: '""',
              position: 'absolute',
              left: 0,
              top: '20%',
              bottom: '20%',
              width: 3,
              borderRadius: '0 3px 3px 0',
              background: `linear-gradient(180deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
              boxShadow: `0 0 12px ${alpha(theme.palette.primary.main, 0.6)}`,
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
        PaperProps={{
          className: "qfd-glass",
          sx: {
            width: drawerWidth,
            borderRight: `1px solid ${alpha('#94a3b8', 0.08)}`,
            backgroundImage: 'none',
          }
        }}
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
          className="qfd-holographic"
          sx={{
            px: 2,
            py: 1.5,
            mb: 4,
            borderRadius: 3,
            cursor: 'pointer',
            border: `1px solid ${alpha('#fff', 0.05)}`,
            '&:hover': { transform: 'scale(1.02)' },
            transition: 'all 0.3s ease',
          }}
          onClick={() => navigate('/')}
        >
          <Box
            sx={{
              width: 36,
              height: 36,
              borderRadius: 2,
              background: theme.palette.financial.qfd.quantum,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: `0 4px 14px ${alpha(theme.palette.financial.qfd.quantum, 0.4)}`,
            }}
          >
            <BrandIcon sx={{ color: '#fff', fontSize: 20 }} />
          </Box>
          <Box>
            <Typography
              variant="h6"
              sx={{
                fontWeight: 900,
                letterSpacing: '-0.04em',
                fontSize: '1.25rem',
                background: theme.palette.financial.qfd.iridescent,
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
                  bgcolor: theme.palette.success.main,
                  boxShadow: `0 0 6px ${alpha(theme.palette.success.main, 0.8)}`,
                  animation: 'live-pulse 1.8s ease-in-out infinite',
                  '@keyframes live-pulse': {
                    '0%, 100%': { opacity: 1 },
                    '50%': { opacity: 0.4 },
                  },
                  flexShrink: 0,
                }}
                className="chip-dot"
              />
            }
            label="MARKET OPEN"
            size="small"
            sx={{
              bgcolor: alpha(theme.palette.success.main, 0.1),
              color: theme.palette.success.main,
              border: `1px solid ${alpha(theme.palette.success.main, 0.2)}`,
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
                {section.items.map((item) => (
                  <NavItem
                    key={item.text}
                    item={item}
                    isActive={location.pathname === item.path}
                    onClick={() => navigate(item.path)}
                  />
                ))}
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
