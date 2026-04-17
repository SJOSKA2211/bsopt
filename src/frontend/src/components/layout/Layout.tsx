import React, { useState } from 'react';
import { useLocation, Link } from 'react-router-dom';
import { Box, Typography, alpha, useMediaQuery, useTheme } from '@mui/material';
import { Dashboard as DashboardIcon, Timeline as MarketIcon, AccountBalanceWallet as PortfolioIcon, Science as ResearchIcon, Warning as RiskIcon, Settings as SettingsIcon } from '@mui/icons-material';
import { AnimatePresence, motion } from 'framer-motion';
import { TickerTape } from '../TickerTape';

const menuItems = [
  { text: 'DASHBOARD', icon: <DashboardIcon sx={{ fontSize: 18 }}/>, path: '/dashboard' },
  { text: 'MARKET_DATA', icon: <MarketIcon sx={{ fontSize: 18 }}/>, path: '/market' },
  { text: 'RESEARCH', icon: <ResearchIcon sx={{ fontSize: 18 }}/>, path: '/research' },
  { text: 'PORTFOLIO', icon: <PortfolioIcon sx={{ fontSize: 18 }}/>, path: '/portfolio' },
  { text: 'RISK_MANIFOLD', icon: <RiskIcon sx={{ fontSize: 18 }}/>, path: '/risk' },
  { text: 'SETTINGS', icon: <SettingsIcon sx={{ fontSize: 18 }}/>, path: '/settings' },
];

export const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const location = useLocation();
  const theme = useTheme();
  
  return (
    <div className="flex w-full h-screen overflow-hidden bg-bento-bg text-white font-sans antialiased">
      <aside className="w-[260px] shrink-0 border-r border-white/5 flex flex-col bg-black/20">
        <Box sx={{ p: 4, mb: 2 }}>
          <Typography variant="h5" className="gradient-text" sx={{ fontWeight: 950, letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: 1.5 }}>
            <div className="w-8 h-8 rounded-lg bg-mint shadow-[0_0_20px_#00ffa3] flex items-center justify-center">
              <div className="w-4 h-4 border-2 border-black rotate-45" />
            </div>
            BS-OPT
          </Typography>
        </Box>
        <div className="px-3 flex-grow space-y-1">
          {menuItems.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <Link key={item.text} to={item.path} className={`flex items-center gap-4 px-4 py-3 rounded-xl transition-all ${isActive ? 'bg-white/5 border border-white/10 text-white' : 'text-white/40 hover:bg-white/5'}`}>
                <span className={isActive ? 'text-mint' : ''}>{item.icon}</span>
                <span className="text-[10px] font-black tracking-widest">{item.text}</span>
                {isActive && <motion.div layoutId="nav-dot" className="ml-auto w-1 h-1 bg-mint rounded-full shadow-[0_0_8px_#00ffa3]" />}
              </Link>
            );
          })}
        </div>
        <Box sx={{ p: 3, borderTop: '1px solid rgba(255,255,255,0.05)' }}>
           <div className="status-pill healthy w-full justify-center">QUANT_NODE_ACTIVE</div>
        </Box>
      </aside>

      <div className="flex-grow flex flex-col h-screen overflow-hidden">
        <header className="shrink-0 z-50 flex flex-col bg-bento-bg/80 backdrop-blur-3xl border-b border-white/5 shadow-2xl">
          <div className="h-14 px-8 flex items-center justify-between">
            <span className="label-secondary opacity-40">TERMINAL_v6.4 // UTC_{new Date().toISOString().slice(11,16)}</span>
            <div className="flex gap-4">
               <div className="status-pill text-[9px] font-black bg-white/5 border border-white/5">LATENCY: 12ms</div>
            </div>
          </div>
          <TickerTape />
        </header>

        <main className="flex-grow overflow-auto relative bg-bento-bg">
           <AnimatePresence mode="wait">
              <motion.div key={location.pathname} initial={{ opacity: 0, scale: 0.99 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 1.01 }} transition={{ duration: 0.25, ease: "easeInOut" }}>
                 {children}
              </motion.div>
           </AnimatePresence>
        </main>
      </div>
    </div>
  );
};
