import React, { useEffect, useState } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { CssBaseline, ThemeProvider, Box, Typography } from '@mui/material';
import mainTheme from './theme/mainTheme';
import useWebSocket from './utils/websocketClient'; 
import apiClient from './utils/apiClient'; // Import logout function

// Assume these components exist and are imported correctly
import Header from './components/layout/Header'; 
import Footer from './components/layout/Footer'; 

import { WebSocketMessagePayload } from './types/websocket';

// Define the context type passed from App.tsx
export interface AppOutletContext {
  isAuthenticated: boolean;
  sendMessage: (message: WebSocketMessagePayload) => void;
  latestMessage: WebSocketMessagePayload | null;
  isConnected: boolean;
}

const App: React.FC = () => {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [loadingAuth, setLoadingAuth] = useState<boolean>(true); // Renamed from loadingAuth

  const navigate = useNavigate();
  const location = useLocation();

  // WebSocket configuration
  const wsUrl = import.meta.env.VITE_WS_URL || `ws://${window.location.host}/ws/frontend_client_${Date.now()}`; 
  const { isConnected, sendMessage, latestMessage } = useWebSocket(wsUrl, {
    onOpen: (_event) => {
      console.log('WebSocket connection opened');
    },
    onMessage: (message: WebSocketMessagePayload) => {
      console.log('Message from server:', message);
      // Process incoming messages here if needed globally
    },
    onClose: (_event) => {
      console.log('WebSocket connection closed:', _event.code, _event.reason);
    },
    onError: (_event) => {
      console.error('WebSocket error:', _event);
    },
    autoConnect: true,
    reconnectInterval: 5000,
    maxReconnectAttempts: 5,
  });

  // Initial authentication check on mount
  useEffect(() => {
    const checkAuth = async () => {
      setLoadingAuth(true);
      const { accessToken, refreshToken } = apiClient.getTokens(); // Use getTokens from apiClient

      if (!accessToken && refreshToken) {
        // Attempt to refresh token if only refresh token exists
        try {
          const response = await apiClient.post('/auth/refresh', { refresh_token: refreshToken });
          const { access_token, refresh_token: newRefreshToken } = response.data;
          apiClient.setTokens(access_token, newRefreshToken); // Use setTokens from apiClient
          setIsAuthenticated(true);
        } catch (error) {
          console.error('Failed to refresh token:', error);
          setIsAuthenticated(false);
          navigate('/login'); 
        }
      } else if (accessToken) {
        // Verify token validity (e.g., by calling a /users/me endpoint)
        try {
          await apiClient.get('/users/me'); 
          setIsAuthenticated(true);
        } catch (error) {
          console.error('Token validation failed:', error);
          setIsAuthenticated(false);
          // If token is invalid, clear stored tokens and redirect
          apiClient.clearTokens(); // Use clearTokens from apiClient
          navigate('/login');
        }
      } else {
        setIsAuthenticated(false);
        // If on a protected route, redirect to login
        const publicPaths = ['/login', '/register', '/forgot-password', '/reset-password/:token'];
        const isPublicPath = publicPaths.some(path => location.pathname === path || location.pathname.startsWith('/reset-password/'));
        
        if (!isPublicPath) {
            navigate('/login', { state: { from: location }, replace: true });
        }
      }
      setLoadingAuth(false);
    };

    checkAuth();
  }, [navigate, location.pathname, location]);

  // --- Logout Functionality ---
  const handleLogout = () => {
    // Call the logout function from apiClient, which handles token clearing and redirection
    // We need to pass 'navigate' to the logout function if it's designed to redirect
    // For now, assuming apiClient.logout handles redirection internally or we handle it here.
    // Let's assume logout returns a promise that resolves after redirection or can be awaited.
    // Or, we can implement the redirection logic here after calling logout.
    
    // For simplicity, assume logout redirects, or handle navigation after clearing tokens:
    apiClient.logout(navigate); // Pass navigate function if needed by logout
  };

  // If still loading authentication, show a loading indicator
  if (loadingAuth) {
    return (
      <ThemeProvider theme={mainTheme}>
        <CssBaseline />
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
          <Typography>Loading application...</Typography>
        </Box>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={mainTheme}>
      <CssBaseline />
      {/* Render Header only if authenticated, or based on route */}
      {/* Assuming Header has logic to show user menu/logout */}
      <Header isAuthenticated={isAuthenticated} onLogout={handleLogout} /> 
      
      <Box sx={{ flexGrow: 1, p: 2, mt: '64px' }}> {/* Main content area, adjusted margin-top for header */}
        <Outlet context={{ isAuthenticated, sendMessage, latestMessage, isConnected }} /> {/* Renders the current route component */}
      </Box>
      
      <Footer />
    </ThemeProvider>
  );
};

export default App;
