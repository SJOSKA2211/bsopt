import { lazy, Suspense } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import { ApolloProvider } from '@apollo/client';
import { apolloClient } from './lib/apollo-client';
import { AnimatePresence, motion } from 'framer-motion';

import { Layout } from './components/layout/Layout';
import SignIn from './components/auth/SignIn';
import { QuantumField } from './components/common/QuantumField';

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
  <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-bento-bg/90 backdrop-blur-2xl">
    <div className="w-[300px] relative">
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: [0, 1, 0.5, 1] }}
        transition={{ duration: 2, repeat: Infinity }}
        className="text-[10px] font-black text-mint font-mono text-center tracking-[0.3em] mb-4 uppercase"
      >
        Syncing_Neural_Manifold
      </motion.div>
      <div className="h-[2px] bg-white/5 rounded-full overflow-hidden relative">
        <motion.div
          initial={{ x: '-100%' }}
          animate={{ x: ['100%', '-100%'] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
          className="absolute inset-0 w-1/2 bg-gradient-to-r from-transparent via-mint to-transparent"
        />
      </div>
    </div>
  </div>
);

const QuantumOverlay = () => (
  <div className="fixed inset-0 pointer-events-none z-[10000] opacity-[0.03] bg-bento-grid bg-[length:40px_40px]" />
);

function AppContent() {
  const location = useLocation();

  return (
    <Layout>
      <AnimatePresence mode="wait">
        <Suspense fallback={<PageLoader />}>
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 12, scale: 0.99 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -12, scale: 1.01 }}
            transition={{ duration: 0.4, ease: [0.23, 1, 0.32, 1] }}
            className="w-full min-h-screen"
          >
            <Routes location={location}>
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
        <QuantumField />
        <QuantumOverlay />
        <BrowserRouter>
          <AppContent />
        </BrowserRouter>
      </QueryClientProvider>
    </ApolloProvider>
  );
}

export default App;