import React, { useState, useEffect } from 'react';
import { useLocation, Link } from 'react-router-dom';
import { 
  Box, 
  Drawer, 
  IconButton, 
  List, 
  ListItem, 
  ListItemIcon, 
  ListItemText,
  Tooltip,
  Typography,
  alpha,
  useMediaQuery,
  useTheme
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Timeline as MarketIcon,
  AccountBalanceWallet as PortfolioIcon,
  Warning as RiskIcon,
  Settings as SettingsIcon,
  Notifications as NotifIcon,
  ExitToApp as LogoutIcon,
  Menu as MenuIcon,
  Science as ResearchIcon,
} from '@mui/icons-material';
import { AnimatePresence, motion } from 'framer-motion';
import { TickerTape } from '../TickerTape';

const drawerWidth = 280;

const menuItems = [
  { text: 'DASHBOARD', icon: <DashboardIcon />, path: '/dashboard' },
  { text: 'MARKET_DATA', icon: <MarketIcon />, path: '/market' },
  { text: 'RESEARCH_LAB', icon: <ResearchIcon />, path: '/research' },
  { text: 'PORTFOLIO_CORE', icon: <PortfolioIcon />, path: '/portfolio' },
  { text: 'RISK_MANIFOLD', icon: <RiskIcon />, path: '/risk' },
  { text: 'SYSTEM_SETTINGS', icon: <SettingsIcon />, path: '/settings' },
];

export const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const theme = useTheme();
  const location = useLocation();
  const isMobile = useMediaQuery(theme.breakpoints.down('lg'));
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    setMobileOpen(false);
  }, [location.pathname]);

  const SidebarContent = () => (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', bgcolor: 'transparent' }}>
      <Box sx={{ p: 4, mb: 2 }}>
        <Typography 
          variant="h5" 
          className="gradient-text"
          sx={{ 
            fontWeight: 950, 
            letterSpacing: '-0.02em',
            fontFamily: 'Outfit',
            display: 'flex',
            alignItems: 'center',
            gap: 1.5
          }}
        >
          <div className="w-8 h-8 rounded-lg bg-mint shadow-[0_0_20px_#00ffa3] flex items-center justify-center">
            <div className="w-4 h-4 border-2 border-black rotate-45" />
          </div>
          BS-OPT
        </Typography>
        <Typography 
          variant="caption" 
          sx={{ 
            color: 'rgba(255,255,255,0.3)', 
            fontWeight: 900, 
            fontSize: '9px',
            letterSpacing: '0.4em',
            mt: 1,
            display: 'block',
            fontFamily: 'Space Grotesk'
          }}
        >
          INSTITUTIONAL_v6.4
        </Typography>
      </Box>

      <List sx={{ px: 2, flex: 1 }}>
        {menuItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <ListItem 
              key={item.text}
              component={Link}
              to={item.path}
              sx={{
                mb: 1,
                borderRadius: '12px',
                cursor: 'pointer',
                bgcolor: isActive ? alpha('#00ffa3', 0.05) : 'transparent',
                border: isActive ? '1px solid rgba(0, 255, 163, 0.15)' : '1px solid transparent',
                transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                textDecoration: 'none',
                color: 'inherit',
                '&:hover': {
                  bgcolor: isActive ? alpha('#00ffa3', 0.08) : 'rgba(255,255,255,0.03)',
                  transform: 'translateX(4px)'
                }
              }}
            >
              <ListItemIcon sx={{ 
                color: isActive ? '#00ffa3' : 'rgba(255,255,255,0.4)',
                minWidth: 42,
                transition: 'color 0.3s'
              }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText 
                primary={item.text} 
                primaryTypographyProps={{
                  sx: { 
                    fontSize: '11px', 
                    fontWeight: isActive ? 900 : 700,
                    letterSpacing: '0.15em',
                    color: isActive ? '#fff' : 'rgba(255,255,255,0.5)',
                    fontFamily: 'Space Grotesk'
                  }
                }}
              />
              {isActive && (
                <motion.div 
                  layoutId="active-marker"
                  className="w-1 h-4 bg-mint rounded-full shadow-[0_0_10px_#00ffa3]"
                />
              )}
            </ListItem>
          );
        })}
      </List>

      <Box sx={{ p: 3, borderTop: '1px solid rgba(255,255,255,0.05)' }}>
        <div className="p-4 rounded-xl bg-white/[0.02] border border-white/5">
          <div className="flex items-center justify-between mb-3">
            <span className="text-[9px] font-black text-white/40 tracking-widest">SERVER_LATENCY</span>
            <span className="text-[10px] font-bold text-mint">12ms</span>
          </div>
          <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
            <div className="h-full w-[15%] bg-mint shadow-[0_0_8px_#00ffa3]" />
          </div>
        </div>
      </Box>
    </Box>
  );

  return (
    <div className="flex w-full h-screen overflow-hidden bg-bento-bg text-white font-sans antialiased">

      {!isMobile && (
        <aside className="w-[280px] shrink-0 border-r border-bento-border bg-bento-bg/50">
          <SidebarContent />
        </aside>
      )}

      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={() => setMobileOpen(false)}
        sx={{
          '& .MuiDrawer-paper': { width: drawerWidth, background: '#050505', backgroundImage: 'none', borderRight: '1px solid rgba(255,255,255,0.1)' },
        }}
      >
        <SidebarContent />
      </Drawer>

      <div className="flex-grow flex flex-col relative min-w-0 h-screen">
        {/* Fixed Header & Ticker System */}
        <header className="shrink-0 z-50 flex flex-col shadow-[0_8px_32px_rgba(0,0,0,0.5)] bg-bento-bg/80 backdrop-blur-2xl">
          <div className="h-[64px] px-6 flex items-center justify-between border-b border-white/5">
            <div className="flex items-center gap-4">
              {isMobile && (
                <Tooltip title="Open menu">
                  <IconButton aria-label="Open menu" onClick={() => setMobileOpen(true)} className="!text-white">
                    <MenuIcon />
                  </IconButton>
                </Tooltip>
              )}
              <div className="flex items-center gap-2">
                 <div className="status-pill bg-white/5 border border-white/5 px-2 py-0.5">
                    <span className="text-[10px] font-black text-white/30 uppercase tracking-[0.2em]">NODE:</span>
                    <span className="text-[10px] font-black text-white uppercase tracking-widest ml-1">QUANT_042</span>
                 </div>
                 <div className="status-pill healthy scale-90">ONLINE</div>
              </div>
            </div>

            <div className="flex items-center gap-1">
               <Tooltip title="Notifications">
                 <IconButton aria-label="Notifications" className="!text-white/40 hover:!text-white transition-colors">
                    <NotifIcon sx={{ fontSize: 20 }} />
                 </IconButton>
               </Tooltip>
               <Tooltip title="Logout">
                 <IconButton aria-label="Logout" className="!text-white/40 hover:!text-white transition-colors">
                    <LogoutIcon sx={{ fontSize: 20 }} />
                 </IconButton>
               </Tooltip>
            </div>
          </div>
          <TickerTape />
        </header>

        {/* Dynamic Content Area */}
        <main className="flex-grow overflow-auto relative bg-bento-bg">
           <AnimatePresence mode="wait">
              <motion.div
                key={location.pathname}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
                className="w-full min-h-0"
              >
                 {children}
              </motion.div>
           </AnimatePresence>
        </main>
      </div>
    </div>
  );
};
