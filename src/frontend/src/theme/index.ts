// src/theme/index.ts – BS-Opt Quantum Financial Deity (QFD) Theme

import { createTheme, alpha } from '@mui/material/styles';
import type { Shadows } from '@mui/material/styles';
import './types.d';

// ============================================================================
// COLOR PALETTE – Quantum Financial Deity System
// ============================================================================

const palette = {
  mode: 'dark' as const,

  primary: {
    main: '#00FFFF',      // Quantum Cyan
    light: '#66FFFF',
    dark: '#00CCCC',
    contrastText: '#020617',
  },

  secondary: {
    main: '#7B68EE',      // Nebula Violet
    light: '#9370DB',
    dark: '#6A5ACD',
    contrastText: '#fff',
  },

  success: {
    main: '#10b981',      // Emerald 500
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
    main: '#38bdf8',      // Sky 400
    light: '#7dd3fc',
    dark: '#0ea5e9',
  },

  background: {
    default: '#020617', // Slate 950
    paper: 'rgba(15, 23, 42, 0.4)',
    elevation1: '#0f172a',
    elevation2: '#1e293b',
    elevation3: '#334155',
  },

  text: {
    primary: '#f8fafc',
    secondary: '#94a3b8',
    disabled: '#64748b',
  },

  divider: alpha('#94a3b8', 0.1),

  financial: {
    bid: '#10b981',
    ask: '#f43f5e',
    positive: '#10b981',
    negative: '#f43f5e',
    neutral: '#94a3b8',
    accents: {
      violet: '#7B68EE',
      amber: '#fbbf24',
      rose: '#f43f5e',
      sky: '#38bdf8',
      emerald: '#10b981',
    },
    greeks: {
      delta: '#00FFFF',
      gamma: '#7B68EE',
      vega: '#D4AF37',   // Electrum Gold
      theta: '#f43f5e',
      rho: '#10b981',
    },
    qfd: {
      nebula: '#7B68EE',
      quantum: '#00FFFF',
      electrum: '#D4AF37',
      iridescent: 'linear-gradient(135deg, #7B68EE 0%, #00FFFF 100%)',
    }
  },
};

// ============================================================================
// TYPOGRAPHY – Outfit for UI + JetBrains Mono for Quanitatives
// ============================================================================

const typography = {
  fontFamily: [
    'Outfit',
    'Inter',
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

  h1: { fontSize: '3rem', fontWeight: 800, letterSpacing: '-0.04em', lineHeight: 1.1 },
  h2: { fontSize: '2.25rem', fontWeight: 800, letterSpacing: '-0.03em', lineHeight: 1.2 },
  h3: { fontSize: '1.875rem', fontWeight: 700, letterSpacing: '-0.02em' },
  h4: { fontSize: '1.5rem', fontWeight: 700, letterSpacing: '-0.01em' },
  h5: { fontSize: '1.25rem', fontWeight: 600 },
  h6: { fontSize: '1.125rem', fontWeight: 600 },

  body1: { fontSize: '1rem', lineHeight: 1.6, letterSpacing: '0.01em' },
  body2: { fontSize: '0.875rem', lineHeight: 1.6, letterSpacing: '0.01em' },

  subtitle1: { fontSize: '1rem', fontWeight: 600, color: '#f8fafc' },
  subtitle2: { fontSize: '0.875rem', fontWeight: 500, color: '#94a3b8' },

  caption: { fontSize: '0.75rem', fontWeight: 600, color: '#64748b', letterSpacing: '0.05em', textTransform: 'uppercase' as const },

  price: {
    fontFamily: '"JetBrains Mono", monospace',
    fontSize: '1.125rem',
    fontWeight: 700,
    letterSpacing: '-0.02em',
  },
  percentage: {
    fontFamily: '"JetBrains Mono", monospace',
    fontSize: '0.875rem',
    fontWeight: 600,
  },
  ticker: {
    fontFamily: '"JetBrains Mono", monospace',
    fontSize: '0.75rem',
    fontWeight: 800,
    letterSpacing: '0.1em',
    textTransform: 'uppercase' as const,
  },
};

// ============================================================================
// COMPONENT OVERRIDES – Quantum Glassmorphism
// ============================================================================

const components = {
  MuiCssBaseline: {
    styleOverrides: {
      '@import': "url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700;800&display=swap')",
      body: {
        backgroundColor: '#020617',
        backgroundImage: `
          radial-gradient(circle at 50% -20%, rgba(123, 104, 238, 0.15) 0%, transparent 80%),
          radial-gradient(circle at 0% 100%, rgba(0, 255, 255, 0.08) 0%, transparent 50%),
          radial-gradient(circle at 100% 100%, rgba(212, 175, 55, 0.05) 0%, transparent 50%)
        `,
        backgroundAttachment: 'fixed',
        color: '#f8fafc',
        '&::-webkit-scrollbar': { width: '6px', height: '6px' },
        '&::-webkit-scrollbar-thumb': { backgroundColor: alpha('#7B68EE', 0.2), borderRadius: '10px' },
        '&::-webkit-scrollbar-track': { backgroundColor: 'transparent' },
      },
    },
  },

  MuiPaper: {
    styleOverrides: {
      root: {
        backgroundImage: 'none',
        backgroundColor: alpha('#0f172a', 0.4),
        backdropFilter: 'blur(40px) saturate(200%)',
        WebkitBackdropFilter: 'blur(40px) saturate(200%)',
        border: `1px solid ${alpha('#f8fafc', 0.08)}`,
        borderRadius: 28,
        boxShadow: `0 30px 60px -12px rgba(0, 0, 0, 0.6)`,
        transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
        '&:hover': {
          borderColor: alpha('#00FFFF', 0.4),
          boxShadow: `0 40px 80px -15px rgba(0, 0, 0, 0.7), 0 0 30px ${alpha('#00FFFF', 0.15)}`,
          transform: 'translateY(-4px)',
        },
      },
    },
  },

  MuiAppBar: {
    styleOverrides: {
      root: {
        backgroundColor: alpha('#020617', 0.8),
        backgroundImage: 'none',
        boxShadow: 'none',
        borderBottom: `0.5px solid ${alpha('#f8fafc', 0.1)}`,
        backdropFilter: 'blur(16px)',
      },
    },
  },

  MuiButton: {
    styleOverrides: {
      root: {
        borderRadius: 14,
        textTransform: 'none' as const,
        fontWeight: 700,
        padding: '10px 24px',
        letterSpacing: '0.02em',
        transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
        '&:hover': {
          transform: 'translateY(-1px)',
          boxShadow: '0 10px 20px -10px rgba(0, 0, 0, 0.5)'
        },
        '&:active': { transform: 'translateY(0)' },
      },
      containedPrimary: {
        background: 'linear-gradient(135deg, #00FFFF 0%, #00CCCC 100%)',
        color: '#020617',
        '&:hover': {
          background: 'linear-gradient(135deg, #66FFFF 0%, #00FFFF 100%)',
          boxShadow: `0 0 25px ${alpha('#00FFFF', 0.4)}`,
        },
      },
      containedSecondary: {
        background: 'linear-gradient(135deg, #7B68EE 0%, #6A5ACD 100%)',
        '&:hover': {
          background: 'linear-gradient(135deg, #9370DB 0%, #7B68EE 100%)',
          boxShadow: `0 0 25px ${alpha('#7B68EE', 0.4)}`,
        },
      },
    },
  },

  MuiTypography: {
    defaultProps: {
      variantMapping: {
        price: 'span',
        percentage: 'span',
        ticker: 'span'
      }
    }
  },

  MuiChip: {
    styleOverrides: {
      root: {
        fontWeight: 700,
        borderRadius: 10,
        fontFamily: '"JetBrains Mono", monospace',
      },
      filledPrimary: {
        background: alpha('#00FFFF', 0.1),
        color: '#00FFFF',
        border: `1px solid ${alpha('#00FFFF', 0.2)}`,
      },
    },
  },

  MuiListItemButton: {
    styleOverrides: {
      root: {
        borderRadius: 14,
        margin: '4px 12px',
        '&.Mui-selected': {
          background: `linear-gradient(90deg, ${alpha('#00FFFF', 0.15)} 0%, transparent 100%)`,
          color: '#00FFFF',
          borderLeft: `3px solid #00FFFF`,
          '& .MuiListItemIcon-root': { color: '#00FFFF' },
          '&:hover': { background: `linear-gradient(90deg, ${alpha('#00FFFF', 0.2)} 0%, transparent 100%)` },
        },
      },
    },
  },
};

const shadows: Shadows = [
  'none',
  '0 2px 4px rgba(0,0,0,0.3)',
  '0 4px 8px rgba(0,0,0,0.3)',
  '0 8px 16px rgba(0,0,0,0.3)',
  ...Array(21).fill('0 25px 50px -12px rgba(0,0,0,0.5)'),
] as Shadows;

export const theme = createTheme({
  palette,
  typography,
  components,
  shadows,
  shape: { borderRadius: 24 },
  spacing: 8,
});