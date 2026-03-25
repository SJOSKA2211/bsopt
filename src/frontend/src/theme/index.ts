// src/theme/index.ts – BS-Opt Institutional Grade Terminal Theme

import { createTheme, alpha } from '@mui/material/styles';
import type { Shadows } from '@mui/material/styles';
import './types.d';

// ============================================================================
// COLOR PALETTE – High Fidelity Quantum System
// ============================================================================

const palette = {
  mode: 'dark' as const,

  primary: {
    main: '#00ffa3', // Neon Emerald
    light: '#6effd1',
    dark: '#00b372',
    contrastText: '#020617',
  },

  secondary: {
    main: '#ff2e7e', // Neon Pink
    light: '#ff70a5',
    dark: '#c20058',
    contrastText: '#020617',
  },

  success: {
    main: '#00ffa3',
    light: '#6effd1',
    dark: '#00b372',
  },

  error: {
    main: '#ff2e7e',
    light: '#ff70a5',
    dark: '#c20058',
  },

  warning: {
    main: '#ffaa00', // Neon Amber
    light: '#ffc14d',
    dark: '#b37700',
  },

  info: {
    main: '#00d4ff', // Neon Sky
    light: '#66e5ff',
    dark: '#0094b3',
  },

  background: {
    default: '#0a0b14', // Deeper space black
    paper: 'rgba(13, 14, 24, 0.7)',
    elevation1: '#11121d',
    elevation2: '#161824',
    elevation3: '#1c1e2e',
  },

  text: {
    primary: '#eff3f8',
    secondary: '#94a3b8',
    disabled: '#475569',
  },

  divider: alpha('#334155', 0.2),

  financial: {
    bid: '#00ffa3',
    ask: '#ff2e7e',
    positive: '#00ffa3',
    negative: '#ff2e7e',
    neutral: '#94a3b8',
    accents: {
      violet: '#a855f7',
      amber: '#ffaa00',
      rose: '#ff2e7e',
      sky: '#00d4ff',
      emerald: '#00ffa3',
    },
    greeks: {
      delta: '#00ffa3',
      gamma: '#00d4ff',
      vega: '#ffaa00',
      theta: '#ff2e7e',
      rho: '#a855f7',
    },
    qfd: {
      emerald: '#00ffa3',
      amber: '#ffaa00',
      sky: '#00d4ff',
      iridescent: 'linear-gradient(135deg, #00ffa3 0%, #00d4ff 100%)',
      quantum: '#00e5ff',
      electrum: '#f1f5f9',
      nebula: '#d8b4fe',
    }
  },
};

// ============================================================================
// TYPOGRAPHY – Modern Sans + Monospace for Data
// ============================================================================

const typography = {
  fontFamily: [
    'Outfit',
    'Inter',
    'system-ui',
    'sans-serif',
  ].join(','),

  fontFamilyMonospace: [
    '"JetBrains Mono"',
    'monospace',
  ].join(','),

  h1: { fontSize: '3.5rem', fontWeight: 900, letterSpacing: '-0.05em', lineHeight: 1.1 },
  h2: { fontSize: '2.5rem', fontWeight: 900, letterSpacing: '-0.04em', lineHeight: 1.2 },
  h3: { fontSize: '2rem', fontWeight: 800, letterSpacing: '-0.03em' },
  h4: { fontSize: '1.5rem', fontWeight: 800, letterSpacing: '-0.02em' },
  h5: { fontSize: '1.25rem', fontWeight: 700 },
  h6: { fontSize: '1.1rem', fontWeight: 700 },

  body1: { fontSize: '1rem', lineHeight: 1.6, letterSpacing: '0.01em' },
  body2: { fontSize: '0.875rem', lineHeight: 1.6, letterSpacing: '0.01em' },

  subtitle1: { fontSize: '1rem', fontWeight: 700, color: '#eff3f8' },
  subtitle2: { fontSize: '0.875rem', fontWeight: 600, color: '#94a3b8' },

  caption: { fontSize: '0.7rem', fontWeight: 800, color: '#64748b', letterSpacing: '0.08em', textTransform: 'uppercase' as const },

  price: {
    fontFamily: '"JetBrains Mono", monospace',
    fontSize: '1.25rem',
    fontWeight: 800,
    letterSpacing: '-0.02em',
  },
  percentage: {
    fontFamily: '"JetBrains Mono", monospace',
    fontSize: '0.9rem',
    fontWeight: 700,
  },
  ticker: {
    fontFamily: '"JetBrains Mono", monospace',
    fontSize: '0.75rem',
    fontWeight: 900,
    letterSpacing: '0.12em',
    textTransform: 'uppercase' as const,
  },
};

// ============================================================================
// COMPONENT OVERRIDES – Institutional Grade Aesthetics
// ============================================================================

const components = {
  MuiCssBaseline: {
    styleOverrides: {
      body: {
        backgroundColor: '#0a0b14',
        backgroundImage: `
          radial-gradient(circle at 50% -20%, rgba(0, 255, 163, 0.12) 0%, transparent 80%),
          radial-gradient(circle at 0% 100%, rgba(0, 212, 255, 0.08) 0%, transparent 50%),
          radial-gradient(circle at 100% 100%, rgba(255, 46, 126, 0.05) 0%, transparent 50%)
        `,
        backgroundAttachment: 'fixed',
        color: '#eff3f8',
      },
    },
  },

  MuiPaper: {
    styleOverrides: {
      root: {
        backgroundImage: 'none',
        backgroundColor: alpha('#11121d', 0.6),
        backdropFilter: 'blur(32px) saturate(180%)',
        WebkitBackdropFilter: 'blur(32px) saturate(180%)',
        border: `1px solid ${alpha('#cbd5e1', 0.08)}`,
        borderRadius: 20,
        boxShadow: `0 20px 40px -12px rgba(0, 0, 0, 0.5)`,
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        '&:hover': {
          borderColor: alpha('#cbd5e1', 0.15),
          boxShadow: `0 30px 60px -15px rgba(0, 0, 0, 0.6)`,
        },
      },
    },
  },

  MuiButton: {
    styleOverrides: {
      root: {
        borderRadius: 8,
        textTransform: 'none' as const,
        fontWeight: 800,
        padding: '12px 28px',
        letterSpacing: '0.02em',
        transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
        '&:active': { transform: 'scale(0.98)' },
      },
      containedPrimary: {
        background: 'linear-gradient(135deg, #00ffa3 0%, #00b372 100%)',
        color: '#020617',
        boxShadow: `0 4px 14px 0 ${alpha('#00ffa3', 0.39)}`,
        '&:hover': {
          background: 'linear-gradient(135deg, #33ffb5 0%, #00ffa3 100%)',
          boxShadow: `0 6px 20px rgba(0, 255, 163, 0.45)`,
          transform: 'translateY(-1px)',
        },
      },
      containedSecondary: {
        background: 'linear-gradient(135deg, #ff2e7e 0%, #c20058 100%)',
        color: '#ffffff',
        boxShadow: `0 4px 14px 0 ${alpha('#ff2e7e', 0.39)}`,
        '&:hover': {
          background: 'linear-gradient(135deg, #ff5c99 0%, #ff2e7e 100%)',
          boxShadow: `0 6px 20px rgba(255, 46, 126, 0.45)`,
          transform: 'translateY(-1px)',
        },
      },
    },
  },

  MuiTab: {
    styleOverrides: {
      root: {
        textTransform: 'none' as const,
        fontWeight: 800,
        fontSize: '0.9rem',
        minHeight: 48,
        transition: 'all 0.2s',
        '&.Mui-selected': {
          color: '#00ffa3',
        },
      },
    },
  },

  MuiTabs: {
    styleOverrides: {
      indicator: {
        height: 3,
        borderRadius: '3px 3px 0 0',
        backgroundColor: '#00ffa3',
        boxShadow: `0 0 12px ${alpha('#00ffa3', 0.8)}`,
      },
    },
  },

  MuiListItemButton: {
    styleOverrides: {
      root: {
        borderRadius: 12,
        margin: '2px 8px',
        transition: 'all 0.2s',
        '&.Mui-selected': {
          backgroundColor: alpha('#00ffa3', 0.1),
          color: '#00ffa3',
          '& .MuiListItemIcon-root': { color: '#00ffa3' },
          '&:hover': { backgroundColor: alpha('#00ffa3', 0.15) },
          '&::after': {
            content: '""',
            position: 'absolute',
            left: 0,
            top: '20%',
            bottom: '20%',
            width: 3,
            backgroundColor: '#00ffa3',
            borderRadius: '0 4px 4px 0',
            boxShadow: `0 0 8px ${alpha('#00ffa3', 0.8)}`,
          },
        },
      },
    },
  },
};

const shadows: Shadows = [
  'none',
  '0 2px 4px rgba(0,0,0,0.3)',
  '0 4px 8px rgba(0,0,0,0.4)',
  '0 8px 16px rgba(0,0,0,0.5)',
  ...Array(21).fill('0 25px 50px -12px rgba(0,0,0,0.6)'),
] as Shadows;

export const theme = createTheme({
  palette,
  typography,
  components,
  shadows,
  shape: { borderRadius: 16 },
  spacing: 8,
});
