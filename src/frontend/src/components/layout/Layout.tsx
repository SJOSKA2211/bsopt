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
  Tooltip,
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
import { stitchTokens } from '../../theme/stitch-tokens';
import { motion, AnimatePresence } from 'framer-motion';

const drawerWidth = 260;

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
      <ListItem disablePadding sx={{ mb: 0.5, px: 2 }}>
        <ListItemButton
          selected={isActive}
          onClick={onClick}
          sx={{
            borderRadius: 0,
            py: 1.2,
            position: 'relative',
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            backgroundColor: isActive ? alpha(item.color || stitchTokens.colors.primary, 0.1) : 'transparent',
            borderLeft: isActive ? `3px solid ${item.color || stitchTokens.colors.primary}` : '3px solid transparent',
            '&:hover': {
              backgroundColor: alpha(item.color || '#fff', 0.05),
              transform: 'translateX(4px)',
              borderLeft: isActive ? `3px solid ${item.color || stitchTokens.colors.primary}` : `3px solid ${alpha(item.color || '#fff', 0.2)}`,
            },
            '&.Mui-selected': {
              backgroundColor: alpha(item.color || stitchTokens.colors.primary, 0.1),
            },
          }}
        >
          <ListItemIcon
            sx={{
              minWidth: 36,
              color: isActive ? (item.color || stitchTokens.colors.primary) : 'rgba(255,255,255,0.4)',
            }}
          >
            <Box sx={{ fontSize: 20, display: 'flex' }}>
              {item.icon}
            </Box>
          </ListItemIcon>
          <ListItemText
            primary={item.text}
            primaryTypographyProps={{
              fontSize: '0.75rem',
              fontWeight: isActive ? 900 : 700,
              color: isActive ? '#fff' : 'rgba(255,255,255,0.5)',
              fontFamily: stitchTokens.typography.labels,
              textTransform: 'uppercase',
              letterSpacing: '0.15em',
            }}
          />
          {isActive && (
            <Box 
              component={motion.div}
              layoutId="nav-active-glow"
              sx={{ 
                position: 'absolute', right: 0, top: '20%', bottom: '20%', width: 2, 
                bgcolor: item.color || stitchTokens.colors.primary,
                boxShadow: `0 0 15px ${item.color || stitchTokens.colors.primary}`
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

  const navItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/', color: stitchTokens.colors.primary },
    { text: 'Trade', icon: <TradeIcon />, path: '/market', color: stitchTokens.colors.secondary },
    { text: 'Optimizer', icon: <AnalysisIcon />, path: '/research', color: '#ff9800' },
    { text: 'Portfolio', icon: <PositionsIcon />, path: '/portfolio', color: stitchTokens.colors.tertiary },
    { text: 'Risk', icon: <AnalysisIcon />, path: '/risk', color: '#E91E63' },
    { text: 'History', icon: <HistoryIcon />, path: '/history', color: '#9e9e9e' },
  ];

  const SidebarContent = () => (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', position: 'relative', overflow: 'hidden' }}>
      {/* Sidebar Dots Layer */}
      <Box className="stitch-dots-container" sx={{ opacity: 0.15 }} />
      
      {/* Brand Shard */}
      <Box sx={{ p: 0, mt: 4, mb: 4, position: 'relative', zIndex: 1 }}>
        <motion.div
           initial={{ x: -50, opacity: 0 }}
           animate={{ x: 0, opacity: 1 }}
           transition={{ duration: 0.8 }}
           className="stitch-slanted-header"
           style={{ width: 'fit-content', padding: '10px 40px 10px 24px' }}
        >
          <Stack direction="row" spacing={1.5} alignItems="center">
            <FlashIcon sx={{ color: '#fff', fontSize: 20 }} />
            <Typography sx={{ fontWeight: 950, fontSize: '1.1rem', letterSpacing: '2px', color: '#fff' }}>
              BS-OPT <Box component="span" sx={{ opacity: 0.5, fontSize: '0.7rem' }}>V2.4</Box>
            </Typography>
          </Stack>
        </motion.div>
      </Box>

      {/* Navigation List */}
      <Box sx={{ flexGrow: 1, position: 'relative', zIndex: 1 }}>
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

      {/* Decorative Shards for Sidebar */}
      <Box className="stitch-abstract-shard float-animation" sx={{ bottom: -30, left: -20, width: 100, height: 100, bgcolor: 'rgba(168, 85, 247, 0.05)', clipPath: stitchTokens.geometry.shard }} />

      {/* System Health / User Block */}
      <Box sx={{ p: 2, borderTop: '1px solid rgba(255,255,255,0.05)', position: 'relative', zIndex: 1, bgcolor: 'rgba(0,0,0,0.2)' }}>
        <Stack spacing={2}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, px: 1 }}>
            <div className="stitch-live-indicator" />
            <Typography className="stitch-label" sx={{ fontSize: '8px', opacity: 0.6 }}>
              RDMA_CORE // LATENCY: 0.4ms
            </Typography>
          </Box>
          
          <Box className="stitch-card" sx={{ p: 1.5, bgcolor: 'rgba(255,255,255,0.02)' }}>
            <Stack direction="row" spacing={1.5} alignItems="center">
              <Avatar 
                sx={{ 
                  width: 32, 
                  height: 32, 
                  bgcolor: alpha(stitchTokens.colors.primary, 0.1),
                  border: `1px solid ${alpha(stitchTokens.colors.primary, 0.3)}`,
                  fontSize: '0.8rem',
                  fontWeight: 900,
                  color: stitchTokens.colors.primary,
                  borderRadius: 0
                }}
              >
                QT
              </Avatar>
              <Box>
                <Typography sx={{ fontSize: '0.7rem', fontWeight: 900, color: '#fff', lineHeight: 1 }}>
                  TRADER_ALPHA
                </Typography>
                <Typography className="stitch-label" sx={{ fontSize: '7px', mt: 0.5, color: stitchTokens.colors.primary }}>
                  INSTITUTIONAL_ACCESS
                </Typography>
              </Box>
              <IconButton size="small" aria-label="Settings" sx={{ ml: 'auto', color: 'rgba(255,255,255,0.2)' }}>
                <SettingsIcon sx={{ fontSize: 14 }} />
              </IconButton>
            </Stack>
          </Box>
        </Stack>
      </Box>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: stitchTokens.colors.background, overflow: 'hidden' }}>
      <CssBaseline />

      {!isMobile && (
        <Box sx={{ width: drawerWidth, flexShrink: 0, borderRight: '1px solid rgba(255,255,255,0.05)', bgcolor: 'rgba(11, 14, 18, 0.8)' }}>
          <SidebarContent />
        </Box>
      )}

      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={() => setMobileOpen(false)}
        ModalProps={{ keepMounted: true }}
        sx={{
          display: { xs: 'block', md: 'none' },
          '& .MuiDrawer-paper': { 
            width: drawerWidth, 
            bgcolor: 'rgba(11, 14, 18, 0.98)',
            backgroundImage: 'none'
          },
        }}
      >
        <SidebarContent />
      </Drawer>

      <Box component="main" sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', position: 'relative' }}>
        {/* Global Abstract Background Decorations */}
        <Box className="stitch-abstract-shard float-animation" sx={{ top: '5%', right: '-10%', width: 500, height: 500, background: 'linear-gradient(135deg, rgba(0, 255, 163, 0.05), transparent)', clipPath: stitchTokens.geometry.shard, filter: 'blur(40px)', zIndex: 0 }} />
        <Box className="stitch-abstract-shard float-animation" sx={{ bottom: '5%', left: '-5%', width: 400, height: 400, background: 'linear-gradient(135deg, rgba(168, 85, 247, 0.04), transparent)', clipPath: stitchTokens.geometry.shard, filter: 'blur(30px)', animationDelay: '-3s', zIndex: 0 }} />

        {/* Top Header */}
        <Box sx={{ 
          height: 64, 
          px: 3, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between',
          borderBottom: '1px solid rgba(255,255,255,0.05)',
          bgcolor: 'rgba(11, 14, 18, 0.8)',
          backdropFilter: 'blur(10px)',
          zIndex: 10
        }}>
          <Stack direction="row" spacing={3} alignItems="center">
            {isMobile && (
              <IconButton onClick={() => setMobileOpen(true)} aria-label="Open sidebar" sx={{ color: '#fff', mr: 1 }}>
                <MenuIcon />
              </IconButton>
            )}
            <Box sx={{ px: 1.5, py: 0.5, bgcolor: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)' }}>
               <Typography sx={{ fontSize: '10px', fontWeight: 900, color: 'rgba(255,255,255,0.4)', fontFamily: stitchTokens.typography.labels }}>
                 TERMINAL_ID: <Box component="span" sx={{ color: '#fff' }}>QUANT_NODE_042</Box>
               </Typography>
            </Box>
            
            {!isMobile && (
              <Stack direction="row" spacing={3} sx={{ ml: 4 }}>
                {[
                  { label: 'CAL_SPREAD', value: '1.45', change: '+0.02' },
                  { label: 'GAMMA_EXP', value: '+1.2M', change: '-5%' },
                ].map(item => (
                  <Box key={item.label}>
                    <Typography className="stitch-label" sx={{ fontSize: '7px', opacity: 0.5 }}>{item.label}</Typography>
                    <Typography className="stitch-mono" sx={{ fontSize: '11px', fontWeight: 800 }}>
                      {item.value} <Box component="span" sx={{ fontSize: '8px', color: item.change.startsWith('+') ? stitchTokens.colors.primary : '#ff2e7e', ml: 0.5 }}>{item.change}</Box>
                    </Typography>
                  </Box>
                ))}
              </Stack>
            )}
          </Stack>

          <Stack direction="row" spacing={2} alignItems="center">
             <Box sx={{ textAlign: 'right', mr: 2 }}>
                <Typography className="stitch-label" sx={{ fontSize: '8px', opacity: 0.5 }}>ACCOUNT_VALUE</Typography>
                <Typography className="stitch-mono" sx={{ fontSize: '14px', fontWeight: 900, color: stitchTokens.colors.primary }}>
                  $2,450,192.40
                </Typography>
             </Box>
             <IconButton sx={{ color: 'rgba(255,255,255,0.4)' }} aria-label="Notifications">
                <NotifIcon fontSize="small" />
             </IconButton>
             <IconButton sx={{ color: 'rgba(255,255,255,0.4)' }} aria-label="Log out">
                <LogoutIcon fontSize="small" />
             </IconButton>
          </Stack>
        </Box>

        {/* Dynamic Viewport */}
        <Box sx={{ flexGrow: 1, p: 0, overflow: 'hidden', position: 'relative', zIndex: 1 }}>
           <AnimatePresence mode="wait">
              <motion.div
                key={location.pathname}
                initial={{ opacity: 0, scale: 0.98, filter: 'blur(10px)' }}
                animate={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
                exit={{ opacity: 0, scale: 1.02, filter: 'blur(10px)' }}
                transition={{ duration: 0.4, ease: [0.23, 1, 0.32, 1] }}
                style={{ height: '100%', width: '100%' }}
              >
                 {children}
              </motion.div>
           </AnimatePresence>
        </Box>
      </Box>
    </Box>
  );
};
