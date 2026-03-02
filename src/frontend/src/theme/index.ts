// src/theme/index.ts – BS-Opt Premium Fintech Theme (Stitch-enhanced)

import { createTheme, alpha } from '@mui/material/styles';
import type { Shadows } from '@mui/material/styles';
import './types.d';

// ============================================================================
// COLOR PALETTE – Deep Navy + Emerald / Sky / Violet fintech system
// ============================================================================

const palette = {
  mode: 'dark' as const,

  primary: {
    main: '#10b981',      // Emerald 500
    light: '#34d399',
    dark: '#059669',
    contrastText: '#fff',
  },

  secondary: {
    main: '#38bdf8',      // Sky 400
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
    default: '#020617',
    paper: 'rgba(15, 23, 42, 0.7)',
    elevation1: '#0f172a',
    elevation2: '#1e293b',
    elevation3: '#334155',
  },

  text: {
    primary: '#f8fafc',
    secondary: '#94a3b8',
    disabled: '#64748b',
  },

  divider: alpha('#94a3b8', 0.08),

  financial: {
    bid: '#10b981',
    ask: '#f43f5e',
    positive: '#10b981',
    negative: '#f43f5e',
    neutral: '#94a3b8',
    accents: {
      violet: '#a855f7',
      amber: '#fbbf24',
      rose: '#f43f5e',
      sky: '#38bdf8',
      emerald: '#10b981',
    },
    greeks: {
      delta: '#38bdf8',
      gamma: '#a855f7',
      vega: '#fbbf24',
      theta: '#f43f5e',
      rho: '#10b981',
    },
  },
};

// ============================================================================
// TYPOGRAPHY – Inter UI + JetBrains Mono for prices
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

  h1: { fontSize: '2.5rem', fontWeight: 800, letterSpacing: '-0.03em' },
  h2: { fontSize: '2rem', fontWeight: 700, letterSpacing: '-0.02em' },
  h3: { fontSize: '1.75rem', fontWeight: 700, letterSpacing: '-0.015em' },
  h4: { fontSize: '1.5rem', fontWeight: 600 },
  h5: { fontSize: '1.25rem', fontWeight: 600 },
  h6: { fontSize: '1rem', fontWeight: 600 },

  body1: { fontSize: '1rem', lineHeight: 1.6 },
  body2: { fontSize: '0.875rem', lineHeight: 1.6 },

  subtitle1: { fontSize: '1rem', fontWeight: 500, color: '#f8fafc' },
  subtitle2: { fontSize: '0.875rem', fontWeight: 500, color: '#94a3b8' },

  caption: { fontSize: '0.75rem', fontWeight: 500, color: '#64748b', letterSpacing: '0.05em' },

  price: {
    fontFamily: '"JetBrains Mono", monospace',
    fontSize: '1.125rem',
    fontWeight: 600,
    letterSpacing: '0.01em',
  },
};

// ============================================================================
// COMPONENT OVERRIDES – Glassmorphism + Premium Fintech
// ============================================================================

const components = {
  MuiCssBaseline: {
    styleOverrides: {
      '@import': "url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap')",
      body: {
        backgroundColor: '#020617',
        backgroundImage: `
          radial-gradient(ellipse at 12% 18%, rgba(16, 185, 129, 0.10) 0%, transparent 50%),
          radial-gradient(ellipse at 88% 8%,  rgba(56, 189, 248, 0.12) 0%, transparent 50%),
          radial-gradient(ellipse at 75% 90%, rgba(168, 85, 247, 0.08) 0%, transparent 50%)
        `,
        backgroundAttachment: 'fixed',
        color: '#f8fafc',
        scrollbarWidth: 'thin',
        scrollbarColor: `${alpha('#94a3b8', 0.18)} transparent`,
        '&::-webkit-scrollbar': { width: '5px', height: '5px' },
        '&::-webkit-scrollbar-thumb': { backgroundColor: alpha('#94a3b8', 0.18), borderRadius: '10px' },
      },
    },
  },

  MuiPaper: {
    styleOverrides: {
      root: {
        backgroundImage: 'none',
        backgroundColor: 'rgba(15, 23, 42, 0.65)',
        backdropFilter: 'blur(20px)',
        WebkitBackdropFilter: 'blur(20px)',
        border: `1px solid ${alpha('#94a3b8', 0.08)}`,
        borderRadius: 20,
        boxShadow: '0 20px 50px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255,255,255,0.05)',
        transition: 'box-shadow 0.25s ease, transform 0.25s ease',
      },
    },
  },

  MuiAppBar: {
    styleOverrides: {
      root: {
        backgroundColor: 'rgba(2, 6, 23, 0.85)',
        backgroundImage: 'none',
        boxShadow: 'none',
        borderBottom: `1px solid ${alpha('#94a3b8', 0.06)}`,
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
      },
    },
  },

  MuiDrawer: {
    styleOverrides: {
      paper: {
        backgroundColor: 'rgba(2, 6, 23, 0.92)',
        backdropFilter: 'blur(24px)',
        WebkitBackdropFilter: 'blur(24px)',
        borderRight: `1px solid ${alpha('#94a3b8', 0.07)}`,
      },
    },
  },

  MuiButton: {
    styleOverrides: {
      root: {
        borderRadius: 12,
        textTransform: 'none' as const,
        fontWeight: 600,
        letterSpacing: '0.01em',
        transition: 'all 0.2s ease',
        '&:hover': { transform: 'translateY(-2px)' },
      },
      containedPrimary: {
        background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
        boxShadow: `0 8px 24px ${alpha('#10b981', 0.3)}`,
        '&:hover': {
          background: 'linear-gradient(135deg, #34d399 0%, #10b981 100%)',
          boxShadow: `0 12px 32px ${alpha('#10b981', 0.45)}`,
        },
      },
    },
  },

  MuiChip: {
    styleOverrides: {
      root: {
        fontWeight: 600,
        fontSize: '0.7rem',
        letterSpacing: '0.05em',
        borderRadius: 8,
      },
      colorSuccess: {
        backgroundColor: alpha('#10b981', 0.15),
        color: '#34d399',
        border: `1px solid ${alpha('#10b981', 0.25)}`,
      },
      colorError: {
        backgroundColor: alpha('#f43f5e', 0.12),
        color: '#fb7185',
        border: `1px solid ${alpha('#f43f5e', 0.22)}`,
      },
      colorWarning: {
        backgroundColor: alpha('#fbbf24', 0.12),
        color: '#fcd34d',
        border: `1px solid ${alpha('#fbbf24', 0.22)}`,
      },
      colorInfo: {
        backgroundColor: alpha('#38bdf8', 0.12),
        color: '#7dd3fc',
        border: `1px solid ${alpha('#38bdf8', 0.22)}`,
      },
    },
  },

  MuiTableRow: {
    styleOverrides: {
      root: {
        transition: 'background 0.15s ease',
        '&:hover': {
          backgroundColor: alpha('#10b981', 0.04),
        },
      },
    },
  },

  MuiListItemButton: {
    styleOverrides: {
      root: {
        borderRadius: 12,
        margin: '2px 8px',
        transition: 'all 0.18s ease',
        '&.Mui-selected': {
          backgroundColor: alpha('#10b981', 0.12),
          color: '#10b981',
          boxShadow: `inset 3px 0 0 ${alpha('#10b981', 0.8)}`,
          '& .MuiListItemIcon-root': { color: '#10b981' },
          '&:hover': { backgroundColor: alpha('#10b981', 0.18) },
        },
        '&:hover': {
          backgroundColor: alpha('#94a3b8', 0.06),
        },
      },
    },
  },

  MuiTooltip: {
    styleOverrides: {
      tooltip: {
        backgroundColor: '#1e293b',
        border: `1px solid ${alpha('#94a3b8', 0.15)}`,
        borderRadius: 10,
        fontSize: '0.75rem',
        backdropFilter: 'blur(12px)',
      },
    },
  },
};

const shadows: Shadows = [
  'none',
  '0 4px 6px -1px rgba(0,0,0,0.3), 0 2px 4px -2px rgba(0,0,0,0.3)',
  '0 10px 15px -3px rgba(0,0,0,0.3), 0 4px 6px -4px rgba(0,0,0,0.3)',
  '0 20px 25px -5px rgba(0,0,0,0.35), 0 8px 10px -6px rgba(0,0,0,0.35)',
  ...Array(21).fill('0 25px 60px -12px rgba(0,0,0,0.45)'),
] as Shadows;

export const theme = createTheme({
  palette,
  typography,
  components,
  shadows,
  shape: { borderRadius: 20 },
  spacing: 8,
});