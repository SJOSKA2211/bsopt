// src/theme/index.ts

import { createTheme, alpha } from '@mui/material/styles';
import type { Shadows } from '@mui/material/styles';
import './types.d';

// ============================================================================
// COLOR PALETTE - Light Financial Dashboard Theme
// ============================================================================

const palette = {
  mode: 'light' as const,

  primary: {
    main: '#2563eb',      // Blue 600 - Primary actions
    light: '#8bf6ff',
    dark: '#1d4ed8',
    contrastText: '#fff',
  },
  
  secondary: {
    main: '#ab47bc',      // Purple 400 - Secondary actions
    light: '#df78ef',
    dark: '#790e8b',
    contrastText: '#fff',
  },
  
  success: {
    main: '#66bb6a',      // Green 400 - Profits, gains
    light: '#98ee99',
    dark: '#338a3e',
  },
  
  error: {
    main: '#ef5350',      // Red 400 - Losses, errors
    light: '#ff867c',
    dark: '#b61827',
  },
  
  warning: {
    main: '#ffa726',      // Orange 400 - Warnings
    light: '#ffd95b',
    dark: '#c77800',
  },
  
  info: {
    main: '#42a5f5',      // Blue 400 - Info
    light: '#80d6ff',
    dark: '#0077c2',
  },
  
  background: {
    default: '#f5f7fb',      // Soft light gray dashboard background
    paper: '#ffffff',        // Cards / surfaces
    elevation1: '#f9fafb',
    elevation2: '#f3f4f6',
    elevation3: '#e5e7eb',
  },
  
  text: {
    primary: '#0f172a',   // Slate 900
    secondary: '#6b7280', // Gray 500
    disabled: '#9ca3af',  // Gray 400
  },
  
  divider: alpha('#9ca3af', 0.12),
  
  // Custom financial colors
  financial: {
    bid: '#66bb6a',       // Green
    ask: '#ef5350',       // Red
    positive: '#66bb6a',
    negative: '#ef5350',
    neutral: '#9ca3af',
    
    // Greeks color scale
    greeks: {
      delta: '#4fc3f7',
      gamma: '#ab47bc',
      vega: '#ffa726',
      theta: '#ef5350',
      rho: '#66bb6a',
    },
  },
};

// ============================================================================
// TYPOGRAPHY - Financial Data Focused
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
  
  // Monospace for numbers
  fontFamilyMonospace: [
    '"JetBrains Mono"',
    'IBM Plex Mono',
    'Monaco',
    'Courier New',
    'monospace',
  ].join(','),
  
  h1: {
    fontSize: '2.5rem',
    fontWeight: 700,
    lineHeight: 1.2,
    letterSpacing: '-0.01562em',
  },
  
  h2: {
    fontSize: '2rem',
    fontWeight: 600,
    lineHeight: 1.3,
  },
  
  h3: {
    fontSize: '1.75rem',
    fontWeight: 600,
    lineHeight: 1.4,
  },
  
  h4: {
    fontSize: '1.5rem',
    fontWeight: 600,
    lineHeight: 1.4,
  },
  
  h5: {
    fontSize: '1.25rem',
    fontWeight: 600,
    lineHeight: 1.5,
  },
  
  h6: {
    fontSize: '1rem',
    fontWeight: 600,
    lineHeight: 1.5,
  },
  
  body1: {
    fontSize: '1rem',
    lineHeight: 1.5,
  },
  
  body2: {
    fontSize: '0.875rem',
    lineHeight: 1.5,
  },
  
  // Custom variants for financial data
  price: {
    fontFamily: '"JetBrains Mono", monospace',
    fontSize: '1.125rem',
    fontWeight: 600,
    letterSpacing: '0.01em',
  },
  
  percentage: {
    fontFamily: '"JetBrains Mono", monospace',
    fontSize: '0.875rem',
    fontWeight: 500,
  },
  
  ticker: {
    fontFamily: '"JetBrains Mono", monospace',
    fontSize: '0.75rem',
    fontWeight: 700,
    letterSpacing: '0.05em',
    textTransform: 'uppercase' as const,
  },
};

// ============================================================================
// COMPONENT OVERRIDES
// ============================================================================

const components = {
  MuiCssBaseline: {
    styleOverrides: {
      '*': {
        margin: 0,
        padding: 0,
        boxSizing: 'border-box',
      },
      'html, body, #root': {
        height: '100%',
        width: '100%',
      },
      body: {
        scrollbarWidth: 'thin',
        scrollbarColor: `rgba(148, 163, 184, 0.8) rgba(249, 250, 251, 1)`,
        '&::-webkit-scrollbar': {
          width: '8px',
          height: '8px',
        },
        '&::-webkit-scrollbar-thumb': {
          backgroundColor: 'rgba(148, 163, 184, 0.8)',
          borderRadius: '4px',
        },
        '&::-webkit-scrollbar-track': {
          backgroundColor: 'rgba(249, 250, 251, 1)',
        },
      },
    },
  },
  
  MuiButton: {
    styleOverrides: {
      root: {
        borderRadius: 12,
        textTransform: 'none' as const,
        fontWeight: 600,
        padding: '10px 20px',
        transition: 'all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease',
        backdropFilter: 'blur(8px)',
        
        '&:hover': {
          transform: 'translateY(-2px) scale(1.02)',
          boxShadow: `0 8px 24px ${alpha(palette.primary.main, 0.4)}`,
        },
      },
      
      sizeLarge: {
        padding: '12px 24px',
        fontSize: '1rem',
      },
      
      sizeSmall: {
        padding: '6px 12px',
        fontSize: '0.875rem',
      },
    },
    
    variants: [
      {
        props: { variant: 'buy' as any },
        style: {
          backgroundColor: palette.success.main,
          color: '#fff',
          '&:hover': {
            backgroundColor: palette.success.dark,
          },
        },
      },
      {
        props: { variant: 'sell' as any },
        style: {
          backgroundColor: palette.error.main,
          color: '#fff',
          '&:hover': {
            backgroundColor: palette.error.dark,
          },
        },
      },
    ],
  },
  
  MuiCard: {
    styleOverrides: {
      root: {
        borderRadius: 16,
        backgroundImage: 'none',
        backgroundColor: palette.background.paper,
        border: `1px solid ${alpha('#e5e7eb', 1)}`,
        boxShadow: `0 18px 45px rgba(15, 23, 42, 0.08)`,
        transition: 'transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease',
        
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: `0 24px 60px rgba(15, 23, 42, 0.12)`,
        },
      },
    },
  },
  
  MuiPaper: {
    styleOverrides: {
      root: {
        backgroundImage: 'none',
        backgroundColor: palette.background.paper,
        border: `1px solid ${alpha('#e5e7eb', 1)}`,
      },
      rounded: {
        borderRadius: 16,
      },
      elevation1: {
        backgroundColor: palette.background.paper,
      },
      elevation2: {
        backgroundColor: palette.background.paper,
      },
      elevation3: {
        backgroundColor: palette.background.paper,
      },
    },
  },
  
  MuiChip: {
    styleOverrides: {
      root: {
        borderRadius: 8,
        fontWeight: 600,
      },
      colorSuccess: {
        backgroundColor: alpha(palette.success.main, 0.15),
        color: palette.success.light,
        border: `1px solid ${alpha(palette.success.main, 0.3)}`,
      },
      colorError: {
        backgroundColor: alpha(palette.error.main, 0.15),
        color: palette.error.light,
        border: `1px solid ${alpha(palette.error.main, 0.3)}`,
      },
    },
  },
  
  MuiDataGrid: {
    styleOverrides: {
      root: {
        border: `1px solid ${alpha(palette.primary.main, 0.1)}`,
        borderRadius: 12,
        
        '& .MuiDataGrid-cell': {
          borderBottom: `1px solid ${alpha(palette.divider, 0.5)}`,
        },
        
        '& .MuiDataGrid-columnHeaders': {
          backgroundColor: palette.background.elevation1,
          borderBottom: `2px solid ${palette.primary.main}`,
        },
        
        '& .MuiDataGrid-row': {
          '&:hover': {
            backgroundColor: alpha(palette.primary.main, 0.05),
          },
          
          '&.Mui-selected': {
            backgroundColor: alpha(palette.primary.main, 0.1),
            
            '&:hover': {
              backgroundColor: alpha(palette.primary.main, 0.15),
            },
          },
        },
      },
    },
  } as any,
  
  MuiTooltip: {
    styleOverrides: {
      tooltip: {
        backgroundColor: palette.background.elevation3,
        border: `1px solid ${alpha(palette.primary.main, 0.3)}`,
        borderRadius: 8,
        padding: '12px 16px',
        fontSize: '0.875rem',
        boxShadow: `0 4px 20px ${alpha('#000', 0.5)}`,
      },
      arrow: {
        color: palette.background.elevation3,
      },
    },
  },
  
  MuiDialog: {
    styleOverrides: {
      paper: {
        borderRadius: 24,
        backgroundImage: 'none',
        backgroundColor: 'rgba(20, 25, 40, 0.75)',
        backdropFilter: 'blur(24px)',
        border: `1px solid rgba(255, 255, 255, 0.1)`,
        boxShadow: `0 24px 64px 0 rgba(0, 0, 0, 0.6)`,
      },
    },
  },
  
  MuiTextField: {
    styleOverrides: {
      root: {
        '& .MuiOutlinedInput-root': {
          borderRadius: 8,
          
          '&:hover .MuiOutlinedInput-notchedOutline': {
            borderColor: alpha(palette.primary.main, 0.5),
          },
          
          '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
            borderColor: palette.primary.main,
            borderWidth: 2,
          },
        },
      },
    },
  },
};

// ============================================================================
// CUSTOM SHADOWS
// ============================================================================

const shadows: Shadows = [
  'none',
  `0 2px 4px ${alpha('#000', 0.1)}`,
  `0 4px 8px ${alpha('#000', 0.15)}`,
  `0 8px 16px ${alpha('#000', 0.2)}`,
  `0 12px 24px ${alpha('#000', 0.25)}`,
  `0 16px 32px ${alpha('#000', 0.3)}`,
  ...Array(19).fill(`0 20px 40px ${alpha('#000', 0.35)}`),
] as Shadows;

// ============================================================================
// SHAPE
// ============================================================================

const shape = {
  borderRadius: 12,
};

// ============================================================================
// SPACING
// ============================================================================

const spacing = 8; // 8px grid system

// ============================================================================
// CREATE THEME
// ============================================================================

export const theme = createTheme({
  palette,
  typography,
  components,
  shadows,
  shape,
  spacing,
});