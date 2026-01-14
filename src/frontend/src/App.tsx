import { lazy, Suspense } from 'react';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Box, CircularProgress } from '@mui/material';
import { theme } from './theme';
import { Layout } from './components/layout/Layout';

// Lazy load pages
const DashboardPage = lazy(() => import('./pages/dashboard/DashboardPage'));
const MarketPage = lazy(() => import('./pages/market/MarketPage'));
const PortfolioPage = lazy(() => import('./pages/portfolio/PortfolioPage'));
const SettingsPage = lazy(() => import('./pages/settings/SettingsPage'));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

const PageLoader = () => (
  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
    <CircularProgress size={60} aria-label="Loading page" />
  </Box>
);

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <BrowserRouter>
          <Layout>
            <Suspense fallback={<PageLoader />}>
              <Routes>
                <Route path="/" element={<DashboardPage />} />
                <Route path="/market" element={<MarketPage />} />
                <Route path="/portfolio" element={<PortfolioPage />} />
                <Route path="/settings" element={<SettingsPage />} />
              </Routes>
            </Suspense>
          </Layout>
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;