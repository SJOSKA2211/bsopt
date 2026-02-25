// src/theme/index.ts

import { createTheme, alpha } from '@mui/material/styles';
import type { Shadows } from '@mui/material/styles';
import './types.d';

// ============================================================================
// COLOR PALETTE - High-Fidelity Dark Financial Theme
// ============================================================================

const palette = {
  mode: 'dark' as const,

  primary: {
    main: '#10b981',      // Emerald 500 - Success/Primary
    light: '#34d399',
    dark: '#059669',
    contrastText: '#fff',
  },
  
  secondary: {
    main: '#38bdf8',      // Sky 400 - Info/Secondary
    light: '#7dd3fc',
    dark: '#0ea5e9',
    contrastText: '#0f172a',
  },
  
  success: {
    main: '#10b981',
    light: '#34d399',
    dark: '#059669',
  },
  
  error: {
    main: '#f43f5e',      // Rose 500
    light: '#fb7185',
    dark: '#e11d48',
  },
  
  warning: {
    main: '#fbbf24',      // Amber 400
    light: '#fcd34d',
    dark: '#f59e0b',
  },
  
  info: {
    main: '#38bdf8',
    light: '#7dd3fc',
    dark: '#0ea5e9',
  },
  
  background: {
    default: '#020617',      // Deep Navy/Black
    paper: 'rgba(15, 23, 42, 0.8)', // Semi-transparent Slate 900
    elevation1: '#0f172a',
    elevation2: '#1e293b',
    elevation3: '#334155',
  },
  
  text: {
    primary: '#f8fafc',   // Slate 50
    secondary: '#94a3b8', // Slate 400
    disabled: '#64748b',  // Slate 500
  },
  
  divider: alpha('#94a3b8', 0.1),
  
  // Custom financial colors
  financial: {
    bid: '#10b981',
    ask: '#f43f5e',
    positive: '#10b981',
    negative: '#f43f5e',
    neutral: '#94a3b8',
    
    // UI Accents from screenshot
    accents: {
      violet: '#a855f7',
      amber: '#f59e0b',
      rose: '#f43f5e',
      sky: '#38bdf8',
      emerald: '#10b981',
    },

    // Greeks color scale
    greeks: {
      delta: '#38bdf8',
      gamma: '#a855f7',
      vega: '#f59e0b',
      theta: '#f43f5e',
      rho: '#10b981',
    },
  },
};

// ============================================================================
// TYPOGRAPHY - Modern Sans & Mono
// ============================================================================

const typography = {
  fontFamily: [
    'Inter',
    'Outfit',
    '-apple-system',
    'BlinkMacSystemFont',
    '"Segoe UI"',
    'Roboto',
    'sans-serif',
  ].join(','),
  
  fontFamilyMonospace: [
    '"JetBrains Mono"',
    'IBM Plex Mono',
    'monospace',
  ].join(','),
  
  h1: { fontSize: '2.5rem', fontWeight: 700, letterSpacing: '-0.02em' },
  h2: { fontSize: '2rem', fontWeight: 600, letterSpacing: '-0.01em' },
  h3: { fontSize: '1.75rem', fontWeight: 600, letterSpacing: '-0.01em' },
  h4: { fontSize: '1.5rem', fontWeight: 600 },
  h5: { fontSize: '1.25rem', fontWeight: 600 },
  h6: { fontSize: '1rem', fontWeight: 600 },
  
  body1: { fontSize: '1rem', lineHeight: 1.6 },
  body2: { fontSize: '0.875rem', lineHeight: 1.6 },
  
  subtitle1: { fontSize: '1rem', fontWeight: 500, color: '#f8fafc' },
  subtitle2: { fontSize: '0.875rem', fontWeight: 500, color: '#94a3b8' },

  caption: { fontSize: '0.75rem', fontWeight: 400, color: '#64748b' },
  
  price: {
    fontFamily: '"JetBrains Mono", monospace',
    fontSize: '1.125rem',
    fontWeight: 600,
    letterSpacing: '0.01em',
  },
};

// ============================================================================
// COMPONENT OVERRIDES - Glassmorphism & Modern UI
// ============================================================================

const components = {
  MuiCssBaseline: {
    styleOverrides: {
      body: {
        backgroundColor: '#020617',
        backgroundImage: `radial-gradient(circle at 15% 20%, rgba(16, 185, 129, 0.08), transparent 45%), 
                         radial-gradient(circle at 85% 10%, rgba(56, 189, 248, 0.1), transparent 45%)`,
        backgroundAttachment: 'fixed',
        color: '#f8fafc',
        scrollbarWidth: 'thin',
        scrollbarColor: `${alpha('#94a3b8', 0.2)} transparent`,
        '&::-webkit-scrollbar': { width: '6px', height: '6px' },
        '&::-webkit-scrollbar-thumb': { backgroundColor: alpha('#94a3b8', 0.2), borderRadius: '10px' },
      },
    },
  },
  
  MuiPaper: {
    styleOverrides: {
      root: {
        backgroundImage: 'none',
        backgroundColor: 'rgba(15, 23, 42, 0.6)',
        backdropFilter: 'blur(16px)',
        border: `1px solid ${alpha('#94a3b8', 0.1)}`,
        borderRadius: 20,
        boxShadow: '0 20px 50px rgba(0,0,0,0.3)',
      },
    },
  },

  MuiAppBar: {
    styleOverrides: {
      root: {
        backgroundColor: 'transparent',
        backgroundImage: 'none',
        boxShadow: 'none',
        borderBottom: `1px solid ${alpha('#94a3b8', 0.05)}`,
        backdropFilter: 'blur(12px)',
      },
    },
  },

  MuiDrawer: {
    styleOverrides: {
      paper: {
        backgroundColor: '#020617',
        borderRight: `1px solid ${alpha('#94a3b8', 0.08)}`,
      },
    },
  },
  
  MuiButton: {
    styleOverrides: {
      root: {
        borderRadius: 12,
        textTransform: 'none' as const,
        fontWeight: 600,
        transition: 'all 0.2s ease-in-out',
        '&:hover': {
          transform: 'translateY(-1px)',
        },
      },
      containedPrimary: {
        boxShadow: `0 8px 20px ${alpha('#10b981', 0.25)}`,
        '&:hover': {
          boxShadow: `0 12px 28px ${alpha('#10b981', 0.4)}`,
        },
      },
    },
  },

  MuiListItemButton: {
    styleOverrides: {
      root: {
        borderRadius: 12,
        margin: '4px 8px',
        '&.Mui-selected': {
          backgroundColor: alpha('#10b981', 0.1),
          color: '#10b981',
          '& .MuiListItemIcon-root': { color: '#10b981' },
          '&:hover': { backgroundColor: alpha('#10b981', 0.15) },
        },
      },
    },
  },
};

const shadows: Shadows = [
  'none',
  '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
  '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)',
  '0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)',
  ...Array(21).fill('0 25px 50px -12px rgb(0 0 0 / 0.25)'),
] as Shadows;

export const theme = createTheme({
  palette,
  typography,
  components,
  shadows,
  shape: { borderRadius: 20 },
  spacing: 8,
});