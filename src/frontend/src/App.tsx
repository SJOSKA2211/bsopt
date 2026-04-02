import { lazy, Suspense } from 'react';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ApolloProvider } from '@apollo/client/react';
import { apolloClient } from './lib/apollo-client';
import { Box, CircularProgress } from '@mui/material';
import { AnimatePresence, motion } from 'framer-motion';
import { useLocation } from 'react-router-dom';
import { theme } from './theme';

import { Layout } from './components/layout/Layout';
import SignIn from './components/auth/SignIn';
import { QuantumField } from './components/common/QuantumField';
import { stitchTokens } from './theme/stitch-tokens';

// Lazy load pages
const DashboardPage = lazy(() => import('./pages/dashboard/DashboardPage'));
const TradeExecutionPage = lazy(() => import('./pages/trading/TradeExecutionPage'));
const StrategyOptimizerPage = lazy(() => import('./pages/research/StrategyOptimizerPage'));
const PortfolioAnalyticsPage = lazy(() => import('./pages/portfolio/PortfolioAnalyticsPage'));
const RiskManagementPage = lazy(() => import('./pages/risk/RiskManagementPage'));
const SettingsPage = lazy(() => import('./pages/settings/SettingsPage'));
const SignUpPage = lazy(() => import('./pages/auth/SignUpPage'));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

const PageLoader = () => (
  <Box
    sx={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      zIndex: 9999,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'rgba(10, 11, 20, 0.9)',
      backdropFilter: 'blur(20px)',
    }}
  >
    <Box sx={{ width: 300, position: 'relative' }}>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: [0, 1, 0.5, 1] }}
        transition={{ duration: 2, repeat: Infinity }}
        style={{
          fontSize: '0.65rem',
          fontWeight: 900,
          color: stitchTokens.colors.primary,
          fontFamily: stitchTokens.typography.data,
          textAlign: 'center',
          letterSpacing: '0.3em',
          marginBottom: 16,
          textTransform: 'uppercase'
        }}
      >
        Synchronizing_Neural_Manifold
      </motion.div>
      <Box sx={{ height: 2, bgcolor: 'rgba(255,255,255,0.05)', borderRadius: 1, overflow: 'hidden', position: 'relative' }}>
        <motion.div
          initial={{ x: '-100%' }}
          animate={{ x: ['100%', '-100%'] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: `linear-gradient(90deg, transparent, ${stitchTokens.colors.primary}, transparent)`,
            width: '50%',
          }}
        />
      </Box>
    </Box>
  </Box>
);

const QuantumOverlay = () => (
  <Box
    sx={{
      position: 'fixed',
      inset: 0,
      pointerEvents: 'none',
      zIndex: 10000,
      opacity: 0.03,
      background: `
        radial-gradient(circle at 2px 2px, ${stitchTokens.colors.primary} 1px, transparent 0)
      `,
      backgroundSize: '40px 40px',
    }}
  />
);

function AppContent() {
  const location = useLocation();

  return (
    <Layout>
      <AnimatePresence mode="wait">
        <Suspense fallback={<PageLoader />}>
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
            style={{ width: '100%' }}
          >
            <Routes location={location} key={location.pathname}>
              <Route path="/" element={<DashboardPage />} />
              <Route path="/market" element={<TradeExecutionPage />} />
              <Route path="/research" element={<StrategyOptimizerPage />} />
              <Route path="/portfolio" element={<PortfolioAnalyticsPage />} />
              <Route path="/positions" element={<PortfolioAnalyticsPage />} />
              <Route path="/risk" element={<RiskManagementPage />} />
              <Route path="/analysis" element={<RiskManagementPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/login" element={<SignIn />} />
              <Route path="/signup" element={<SignUpPage />} />
            </Routes>
          </motion.div>
        </Suspense>
      </AnimatePresence>
    </Layout>
  );
}

function App() {
  return (
    <ApolloProvider client={apolloClient}>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <QuantumField />
          <QuantumOverlay />
          <BrowserRouter>
            <AppContent />
          </BrowserRouter>
        </ThemeProvider>
      </QueryClientProvider>
    </ApolloProvider>
  );
}

export default App;