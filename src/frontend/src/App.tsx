import { lazy, Suspense } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, Routes, Route, useLocation, Navigate } from 'react-router-dom';
import { ApolloProvider } from '@apollo/client/react';
import { apolloClient } from './lib/apollo-client';
import { ThemeProvider } from '@mui/material/styles';
import { theme } from './theme';
import { CssBaseline } from '@mui/material';

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
    <div className="w-[120px] h-[1px] bg-white/5 rounded-full overflow-hidden relative">
      <div className="absolute inset-0 w-1/2 bg-mint animate-[loading-slide_1.5s_infinite_ease-in-out]" />
    </div>
  </div>
);

function AppContent() {
  const location = useLocation();
  const isAuthPage = location.pathname === '/login' || location.pathname === '/signup';

  const routes = (
    <Suspense fallback={<PageLoader />}>
      <Routes location={location}>
        {/* Redirect base / to /dashboard */}
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        
        {/* Core Pages */}
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/market" element={<TradeExecutionPage />} />
        <Route path="/research" element={<StrategyOptimizerPage />} />
        <Route path="/portfolio" element={<PortfolioAnalyticsPage />} />
        <Route path="/positions" element={<Navigate to="/portfolio" replace />} />
        <Route path="/risk" element={<RiskManagementPage />} />
        <Route path="/analysis" element={<Navigate to="/risk" replace />} />
        <Route path="/settings" element={<SettingsPage />} />
        
        {/* Auth Pages */}
        <Route path="/login" element={<SignIn />} />
        <Route path="/signup" element={<SignUpPage />} />
      </Routes>
    </Suspense>
  );

  // Layout contains its own AnimatePresence and motion.div for page transitions
  return isAuthPage ? routes : <Layout>{routes}</Layout>;
}

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ApolloProvider client={apolloClient}>
        <QueryClientProvider client={queryClient}>
          <BrowserRouter>
            <AppContent />
          </BrowserRouter>
        </QueryClientProvider>
      </ApolloProvider>
    </ThemeProvider>
  );
}

export default App;