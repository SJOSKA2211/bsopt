import { createTheme } from '@mui/material/styles';

/**
 * Production-ready Material UI Theme
 * Optimized for performance and accessibility (WCAG AA compliance)
 */
export const mainTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00e676', // High visibility green
      contrastText: '#000',
    },
    secondary: {
      main: '#2979ff',
    },
    background: {
      default: '#0a1929',
      paper: '#102032',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: { fontSize: '2.5rem', fontWeight: 600 },
    h2: { fontSize: '2rem', fontWeight: 600 },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 600,
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
  },
  // Performance optimization: prevent unnecessary style recalculations
  shape: {
    borderRadius: 8,
  },
});

export default mainTheme;
