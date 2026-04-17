import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { 
  Box, 
  Drawer, 
  IconButton, 
  List, 
  ListItem, 
  ListItemIcon, 
  ListItemText,
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
  { text: 'RESEARCH_LAB', icon: <ResearchIcon />, path: '/market/research' },
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
              onClick={() => {}}
              sx={{
                mb: 1,
                borderRadius: '12px',
                cursor: 'pointer',
                bgcolor: isActive ? alpha('#00ffa3', 0.05) : 'transparent',
                border: isActive ? '1px solid rgba(0, 255, 163, 0.15)' : '1px solid transparent',
                transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
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
                  layoutId="active-pill"
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
    <div className="flex min-h-screen bg-bento-bg text-white font-sans antialiased">

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
          '& .MuiDrawer-paper': { width: drawerWidth, background: '#050505', backgroundImage: 'none' },
        }}
      >
        <SidebarContent />
      </Drawer>

      <main className="flex-grow flex flex-col relative min-w-0 h-screen overflow-hidden">
        {/* Fixed Header & Ticker System */}
        <div className="shrink-0 flex flex-col z-50 shadow-[0_4px_30px_rgba(0,0,0,0.5)]">
          <header className="h-[72px] px-6 flex items-center justify-between border-b border-bento-border bg-bento-bg/60 backdrop-blur-xl">
            <div className="flex items-center gap-4">
              {isMobile && (
                <IconButton onClick={() => setMobileOpen(true)} className="!text-white">
                  <MenuIcon />
                </IconButton>
              )}
              <div className="status-pill bg-white/5 border border-bento-border px-3 py-1.5">
                 <span className="text-[11px] font-bold text-white/40 uppercase tracking-widest">
                   NODE: <span className="text-white">QUANT_042</span>
                 </span>
              </div>
            </div>

            <div className="flex items-center gap-1">
               <IconButton className="!text-white/40 hover:!text-white transition-colors">
                  <NotifIcon />
               </IconButton>
               <IconButton className="!text-white/40 hover:!text-white transition-colors">
                  <LogoutIcon />
               </IconButton>
            </div>
          </header>
          <TickerTape />
        </div>

        <div className="flex-grow flex flex-col min-h-0 overflow-auto">
           <AnimatePresence mode="wait">
              <motion.div
                key={location.pathname}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.3 }}
                className="flex-grow"
              >
                 {children}
              </motion.div>
           </AnimatePresence>
        </div>
      </main>
    </div>
  );
};
