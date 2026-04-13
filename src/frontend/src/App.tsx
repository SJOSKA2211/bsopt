import { lazy, Suspense } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import { ApolloProvider } from '@apollo/client/react';
import { apolloClient } from './lib/apollo-client';
import { AnimatePresence, motion } from 'framer-motion';

import { Layout } from './components/layout/Layout';
import SignIn from './components/auth/SignIn';

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
  <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-zinc-950/90 backdrop-blur-xl">
    <div className="w-[200px] h-[2px] bg-white/5 rounded-full overflow-hidden relative">
      <motion.div
        initial={{ x: '-100%' }}
        animate={{ x: '100%' }}
        transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
        className="absolute inset-0 w-1/2 bg-blue-500"
      />
    </div>
  </div>
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
            transition={{ duration: 0.2 }}
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
        <BrowserRouter>
          <AppContent />
        </BrowserRouter>
      </QueryClientProvider>
    </ApolloProvider>
  );
}

export default App;