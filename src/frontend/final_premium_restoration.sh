#!/bin/bash
ROOT="/home/kamau/bsopt/src/frontend"

# 1. Premium Layout.tsx
rm -f "$ROOT/src/components/layout/Layout.tsx"
cat > "$ROOT/src/components/layout/Layout.tsx" <<'EOF'
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
EOF

# 2. Premium TickerTape.tsx
rm -f "$ROOT/src/components/TickerTape.tsx"
cat > "$ROOT/src/components/TickerTape.tsx" <<'EOF'
import React, { useMemo } from 'react';
import { Box, Typography, alpha } from '@mui/material';
import { motion } from 'framer-motion';

export const TickerTape: React.FC = () => {
  const displayItems = useMemo(() => [
    { symbol: 'SPY', price: '512.42', percentChange: '+0.45%', up: true },
    { symbol: 'QQQ', price: '445.12', percentChange: '-0.12%', up: false },
    { symbol: 'BTC/USD', price: '64,120.42', percentChange: '+2.14%', up: true },
    { symbol: 'ETH/USD', price: '3,420.12', percentChange: '+1.85%', up: true },
    { symbol: 'NVDA', price: '894.22', percentChange: '+4.12%', up: true },
    { symbol: 'SPY', price: '512.42', percentChange: '+0.45%', up: true },
    { symbol: 'QQQ', price: '445.12', percentChange: '-0.12%', up: false },
    { symbol: 'BTC/USD', price: '64,120.42', percentChange: '+2.14%', up: true },
  ], []);

  return (
    <Box sx={{ width: '100%', height: 32, bgcolor: 'rgba(0,0,0,0.5)', borderBottom: '1px solid rgba(255,255,255,0.03)', overflow: 'hidden', position: 'relative', display: 'flex', alignItems: 'center' }}>
      <motion.div animate={{ x: [0, -1200] }} transition={{ x: { repeat: Infinity, duration: 40, ease: "linear" } }} style={{ display: 'flex', alignItems: 'center', gap: '48px', paddingLeft: '48px', whiteSpace: 'nowrap' }}>
        {displayItems.map((t, i) => (
          <Box key={i} sx={{ display: 'flex', alignItems: 'center', gap: 1.5, opacity: 0.8 }}>
            <Typography sx={{ fontWeight: 950, color: '#fff', fontSize: '9px', letterSpacing: '0.1em' }}>{t.symbol}</Typography>
            <Typography className="data-mono" sx={{ fontSize: '10px', color: 'rgba(255,255,255,0.5)' }}>{t.price}</Typography>
            <Box sx={{ px: 0.8, py: 0.1, borderRadius: '4px', bgcolor: alpha(t.up ? '#00ffa3' : '#ef4444', 0.1), border: `1px solid ${alpha(t.up ? '#00ffa3' : '#ef4444', 0.2)}` }}>
               <Typography sx={{ fontSize: '8px', color: t.up ? '#00ffa3' : '#ef4444', fontWeight: 900 }}>{t.up ? '▲' : '▼'} {t.percentChange}</Typography>
            </Box>
          </Box>
        ))}
      </motion.div>
    </Box>
  );
};
EOF

# 3. Premium DashboardPage.tsx
rm -f "$ROOT/src/pages/dashboard/DashboardPage.tsx"
cat > "$ROOT/src/pages/dashboard/DashboardPage.tsx" <<'EOF'
import React from 'react';
import { motion } from 'framer-motion';

const KpiCard = ({ label, value, color, prefix = '', index = 0 }: any) => (
  <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.05 }} className="bento-card relative overflow-hidden group">
     <div className="absolute top-0 right-0 w-16 h-16 bg-gradient-to-bl from-white/5 to-transparent pointer-events-none" />
     <span className="label-secondary opacity-60">{label}</span>
     <div className="flex items-baseline gap-1 mt-2">
        <span className="data-mono text-3xl font-black text-white group-hover:text-mint transition-colors">{prefix}{value}</span>
     </div>
     <div className="h-0.5 w-12 mt-4 rounded-full shadow-[0_0_10px_currentcolor]" style={{ backgroundColor: color, color }} />
  </motion.div>
);

const DashboardPage = () => (
  <div className="p-8 space-y-8 min-h-full">
    <div className="bento-grid">
       <div className="col-span-12 sm:col-span-6 lg:col-span-3">
          <KpiCard label="SYSTEM_GAMMA" value="2.412" color="#00FFA3" index={0} />
       </div>
       <div className="col-span-12 sm:col-span-6 lg:col-span-3">
          <KpiCard label="PORTFOLIO_NAV" value="254,120.42" prefix="$" color="#F59E0B" index={1} />
       </div>
       <div className="col-span-12 sm:col-span-6 lg:col-span-3">
          <KpiCard label="VEGA_SENS" value="4.12k" color="#14B8A6" index={2} />
       </div>
       <div className="col-span-12 sm:col-span-6 lg:col-span-3">
          <KpiCard label="MODEL_CONFIDENCE" value="98.4%" color="#BD00FF" index={3} />
       </div>

       <div className="col-span-12 lg:col-span-8">
          <div className="bento-card h-[520px] flex flex-col relative overflow-hidden group">
             <div className="p-4 border-b border-white/5 mb-8 flex justify-between items-center bg-white/[0.02]">
                <span className="label-secondary opacity-40">DEEP_INFERENCE_ENGINE // RT_PROBABILITY_DENSITY</span>
                <span className="status-pill text-[9px] healthy scale-90">COMPUTING</span>
             </div>
             <div className="flex-grow flex flex-col items-center justify-center opacity-10">
                <div className="w-32 h-32 border-2 border-mint rounded-full animate-pulse flex items-center justify-center">
                   <div className="w-16 h-16 border border-mint/40 rounded-full animate-ping" />
                </div>
                <span className="mt-8 text-[11px] font-black tracking-[1.5em] uppercase text-mint">SYSTEM_ACTIVE</span>
             </div>
          </div>
       </div>

       <div className="col-span-12 lg:col-span-4">
          <div className="bento-card h-[520px] !p-0 overflow-hidden border-white/5">
             <div className="p-6 border-b border-white/5 bg-white/[0.02]">
                <span className="label-secondary opacity-40">RISK_CONCENTRATION_MAP</span>
             </div>
             <div className="p-8 space-y-8">
                {[
                  { l: "EQUITY_VOL", v: 85, c: "#00FFA3" },
                  { l: "CREDIT_SPREAD", v: 42, c: "#F59E0B" },
                  { l: "FX_VARIANCE", v: 28, c: "#BD00FF" }
                ].map((r, i) => (
                  <div key={i} className="space-y-3">
                     <div className="flex justify-between text-[9px] font-black tracking-widest text-white/40 uppercase">
                        <span>{r.l}</span>
                        <span style={{ color: r.c }}>{r.v}%</span>
                     </div>
                     <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                        <motion.div initial={{ width: 0 }} animate={{ width: `${r.v}%` }} transition={{ duration: 1, delay: i * 0.1 }} className="h-full shadow-[0_0_10px_currentcolor]" style={{ backgroundColor: r.c, color: r.c }} />
                     </div>
                  </div>
                ))}
             </div>
          </div>
       </div>
    </div>
  </div>
);
export default DashboardPage;
EOF

echo "FINAL PREMIUM RESTORATION COMPLETE."
