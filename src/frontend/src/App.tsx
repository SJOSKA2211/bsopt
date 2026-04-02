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
  <Box sx={{ position: 'fixed', top: 0, left: 0, right: 0, zIndex: 9999 }}>
    <motion.div
      initial={{ scaleX: 0, originX: 0 }}
      animate={{ scaleX: [0, 0.4, 0.7, 0.9, 1] }}
      transition={{ 
        duration: 2, 
        ease: "easeInOut",
        times: [0, 0.2, 0.5, 0.8, 1],
        repeat: Infinity,
        repeatDelay: 0.2
      }}
      style={{ 
        height: 3, 
        background: `linear-gradient(90deg, ${stitchTokens.colors.primary} 0%, ${stitchTokens.colors.secondary} 100%)`, 
        boxShadow: `0 0 10px ${stitchTokens.colors.primary}`
      }}
    />
  </Box>
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
          <BrowserRouter>
            <AppContent />
          </BrowserRouter>
        </ThemeProvider>
      </QueryClientProvider>
    </ApolloProvider>
  );
}

export default App;