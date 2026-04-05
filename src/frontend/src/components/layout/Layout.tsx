import React, { useState } from 'react';
import { 
  CssBaseline, 
  Drawer, 
  Avatar, 
  useMediaQuery,
  useTheme
} from '@mui/material';
import { Zap as FlashIcon, Globe as DashboardIcon, Layers as PositionsIcon, TrendingUp as TradeIcon } from '../common/Icons';

// Simple SVG placeholders for missing layout icons to avoid pulling in an entire library
const AnalysisIcon = () => <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 3v18h18"/><path d="M18 9l-5 5-4-4-5 5"/></svg>;
const SettingsIcon = () => <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>;
const NotifIcon = () => <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>;
const LogoutIcon = () => <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>;
const MenuIcon = () => <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>;
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
      className="px-4 mb-1"
    >
      <button
        onClick={onClick}
        className={`w-full flex items-center gap-4 px-4 py-3.5 rounded-xl transition-all duration-300 relative group
          ${isActive ? 'bg-white/5 text-white' : 'text-white/40 hover:bg-white/10 hover:text-white/60 hover:translate-x-1'}
        `}
      >
        <div className={`text-[22px] flex items-center ${isActive ? 'text-mint' : 'text-current'}`}>
          {item.icon}
        </div>
        <span className={`text-[13px] tracking-wide ${isActive ? 'font-bold' : 'font-medium'}`}>
          {item.text}
        </span>
        {isActive && (
          <motion.div
            layoutId="nav-glow"
            className="absolute right-0 w-1 h-5 bg-mint rounded-l-full shadow-[0_0_10px_#00FFA3]"
          />
        )}
      </button>
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
    <div className="h-full flex flex-col p-6 overflow-hidden">
      {/* Brand */}
      <div className="flex items-center gap-3 mb-12 px-2">
        <FlashIcon className="text-mint text-2xl" />
        <h1 className="text-lg font-black tracking-tight text-white uppercase">
          BSOPT_V2
        </h1>
      </div>

      {/* Navigation */}
      <nav className="flex-grow space-y-1">
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
      </nav>

      {/* User & Health */}
      <AnimatedCard className="mt-auto !p-4 !bg-white/5 border-white/5">
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-3">
            <Avatar className="!w-8 !h-8 !bg-white/10 !text-[12px] font-bold">QT</Avatar>
            <div className="flex flex-col">
              <span className="text-[12px] font-bold text-white">Trader_Alpha</span>
              <span className="text-[10px] text-white/40 font-medium uppercase tracking-wider">Inst_Access</span>
            </div>
          </div>
          <div className="p-3 bg-black/40 rounded-xl border border-white/5 flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-mint animate-pulse shadow-[0_0_8px_#00FFA3]" />
            <span className="text-[10px] font-black text-mint uppercase tracking-tight">
              RT_FEED: {health.latency}ms
            </span>
          </div>
        </div>
      </AnimatedCard>
    </div>
  );

  return (
    <div className="flex min-h-screen bg-bento-bg text-white font-sans antialiased">
      <CssBaseline />

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

      <main className="flex-grow flex flex-col relative min-w-0">
        {/* Header */}
        <header className="h-[72px] px-6 flex items-center justify-between border-b border-bento-border bg-bento-bg/60 backdrop-blur-xl sticky top-0 z-50">
          <div className="flex items-center gap-4">
            {isMobile && (
              <button aria-label="Open navigation menu" onClick={() => setMobileOpen(true)} className="p-2 !text-white hover:bg-white/10 rounded-full transition-colors">
                <MenuIcon />
              </button>
            )}
            <div className="status-pill bg-white/5 border border-bento-border px-3 py-1.5">
               <span className="text-[11px] font-bold text-white/40 uppercase tracking-widest">
                 NODE: <span className="text-white">QUANT_042</span>
               </span>
            </div>
          </div>

          <div className="flex items-center gap-1">
             <button aria-label="View notifications" className="p-2 !text-white/40 hover:!text-white hover:bg-white/10 rounded-full transition-colors">
                <NotifIcon />
             </button>
             <button aria-label="Logout" className="p-2 !text-white/40 hover:!text-white hover:bg-white/10 rounded-full transition-colors">
                <LogoutIcon />
             </button>
          </div>
        </header>

        <TickerTape />

        <div className="flex-grow flex flex-col min-h-0">
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
