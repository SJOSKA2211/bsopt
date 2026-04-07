import React, { useState } from 'react';
import { 
  CssBaseline, 
  Drawer, 
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
              <IconButton onClick={() => setMobileOpen(true)} className="!text-white" aria-label="Open menu">
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
             <IconButton className="!text-white/40 hover:!text-white transition-colors" aria-label="View notifications">
                <NotifIcon />
             </IconButton>
             <IconButton className="!text-white/40 hover:!text-white transition-colors" aria-label="Log out">
                <LogoutIcon />
             </IconButton>
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
