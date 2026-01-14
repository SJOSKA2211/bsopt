import React, { useState, useEffect } from 'react';
import {
  useNavigate, useOutletContext
} from 'react-router-dom';
import {
  Container,
  Typography,
  Grid,
  Paper,
  Box,
  Alert,
  Button,
  CircularProgress,
} from '@mui/material';
import LazyPlot from '@/components/charts/LazyPlot';
import LazyComponent from '@/components/common/LazyComponent';
import apiClient from '@/utils/apiClient';
import { AxiosError } from 'axios'; 
import { Data } from 'plotly.js'; // Import Data type from plotly.js

import { AppOutletContext } from '../App';
import {
  UserStats,
  PriceUpdatePayload,
  UserProfileUpdatePayload,
  UserStatusUpdatePayload,
  WebSocketMessagePayload,
} from '../types/websocket';

interface ErrorResponse {
  message?: string;
  detail?: string | unknown;
}

const Dashboard: React.FC = () => {
  const [userStats, setUserStats] = useState<UserStats | null>(null);
  const [userProfile, setUserProfile] = useState<UserProfileUpdatePayload | null>(null); // State for user profile
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [latestWsMessage, setLatestWsMessage] = useState<WebSocketMessagePayload | null>(null); // State to hold processed WS messages
  const [realtimePrices, setRealtimePrices] = useState<Record<string, PriceUpdatePayload>>({}); // Added missing state

  const { isAuthenticated, sendMessage, latestMessage: rawLatestMessage, isConnected } = useOutletContext<AppOutletContext>();
  const navigate = useNavigate();

  // Fetch initial user stats and profile
  useEffect(() => {
    const fetchUserData = async () => {
      if (!isAuthenticated) {
        navigate('/login');
        return;
      }
      
      setLoading(true);
      setError(null); 
      try {
        // Fetch stats
        const statsResponse = await apiClient.get('/users/me/stats');
        setUserStats(statsResponse.data);

        // Fetch profile
        const profileResponse = await apiClient.get('/users/me');
        setUserProfile(profileResponse.data);

      } catch (err: unknown) { // Changed from AxiosError to unknown
        console.error('Failed to fetch user data:', err);
        let displayMessage: string | null = null;
        if (typeof err === 'object' && err !== null && 'response' in err) {
          const axiosError = err as AxiosError;
          const errorData = axiosError.response?.data as ErrorResponse;
          if (axiosError.response?.status === 401 || axiosError.response?.status === 403) {
            displayMessage = 'Authentication failed. Please log in again.';
            localStorage.removeItem('accessToken');
            localStorage.removeItem('refreshToken');
            navigate('/login');
          } else if (typeof errorData.message === 'string') {
            displayMessage = errorData.message;
          } else if (typeof errorData.detail === 'string') {
            displayMessage = errorData.detail;
          }
        }
        setError(displayMessage || 'An unexpected error occurred. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchUserData(); // Corrected function name
  }, [isAuthenticated, navigate]);

  // Process incoming WebSocket messages
  useEffect(() => {
    if (rawLatestMessage) {
      try {
        const message: WebSocketMessagePayload = typeof rawLatestMessage === 'string' ? JSON.parse(rawLatestMessage) : rawLatestMessage;
        
        console.log('Dashboard processed WebSocket message:', message);
        setLatestWsMessage(message); // Store processed message

        switch (message.type) {
          case 'user_stats_update':
            // Use type guard to safely access payload
            if ('payload' in message && typeof message.payload === 'object' && message.payload !== null) {
              setUserStats(prevStats => {
                if (!prevStats) return message.payload as UserStats;
                return { ...prevStats, ...(message.payload as UserStats) };
              });
            }
            console.log('User stats updated via WebSocket.');
            break;
          case 'price_update': {
            // Use type guard to safely access payload
            if ('payload' in message && typeof message.payload === 'object' && message.payload !== null) {
              const priceData: PriceUpdatePayload = message.payload as PriceUpdatePayload;
              // Update a specific price or manage a list of real-time prices
              // For simplicity, we'll just log it and add to a displayed list
              setRealtimePrices(prevPrices => {
                  const updatedPrices = { ...prevPrices };
                  // Only update if this instrument is relevant or show all for demo
                  updatedPrices[priceData.instrument.id] = priceData;
                  // Limit the number of displayed prices to avoid performance issues
                  const priceIds = Object.keys(updatedPrices);
                  if (priceIds.length > 10) { // Keep only last 10 updates
                      const oldestKey = priceIds[0];
                      delete updatedPrices[oldestKey];
                  }
                  return updatedPrices;
              });
              console.log('Received price update:', priceData);
            }
            break;
          }
          case 'batch_price_update':
             console.log('Received batch price update:', message.payload);
             // Handle batch updates if needed
             break;
          case 'system_stats_update':
            console.log('Received system stats update:', message.payload);
            break;
          case 'user_profile_update': {
            // Use type guard to safely access payload
            if ('payload' in message && typeof message.payload === 'object' && message.payload !== null) {
              const profileUpdate: UserProfileUpdatePayload = message.payload as UserProfileUpdatePayload;
              setUserProfile(profileUpdate); // Update user profile directly
              console.log('User profile updated via WebSocket:', profileUpdate);
            }
            break;
          }
          case 'user_status_update': {
            // Use type guard to safely access payload
            if ('payload' in message && typeof message.payload === 'object' && message.payload !== null) {
              const statusUpdate: UserStatusUpdatePayload = message.payload as UserStatusUpdatePayload;
              // If it's our user, update status or potentially redirect if deactivated
              if (userProfile && userProfile.id === statusUpdate.user_id) {
                setUserProfile(prev => prev ? { ...prev, is_active: statusUpdate.is_active, status: statusUpdate.status } : null);
                if (!statusUpdate.is_active) {
                  alert("Your account has been deactivated. You will be logged out.");
                  localStorage.removeItem('accessToken');
                  localStorage.removeItem('refreshToken');
                  navigate('/login');
                }
              }
              console.log('User status updated via WebSocket:', statusUpdate);
            }
            break;
          }
        }
      } catch (e) {
        console.error('Failed to process WebSocket message:', e);
      }
    }
  }, [rawLatestMessage, navigate, userProfile]); // Add navigate and userProfile to dependencies

  // Mock data for charts
  const mockChartData: Partial<Data>[] = [
    { x: [1, 2, 3], y: [2, 6, 3], type: 'scatter', mode: 'lines+markers', marker: { color: 'red' } },
  ];
  const mockChartLayout = { title: 'Sample Market Overview' };

  // Function to send a test message via WebSocket
  const handleSendMessage = () => {
    const testMessage = {
      type: 'client_interaction', 
      payload: { action: 'dashboard_button_click', timestamp: new Date().toISOString() }
    };
    sendMessage(testMessage);
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2, height: 400, display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" gutterBottom>Market Overview</Typography>
            <LazyPlot
              title="Sample Market Data"
              data={mockChartData}
              layout={{ ...mockChartLayout, autosize: true }}
              height="calc(100% - 40px)" 
            />
          </Paper>
        </Grid>
        <Grid item xs={12} md={4}>
          <LazyComponent height={300}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6">User Statistics</Typography>
              {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                  <CircularProgress size={24} />
                  <Typography sx={{ ml: 2 }}>Loading stats...</Typography>
                </Box>
              ) : error ? (
                <Typography color="error">Could not load stats.</Typography>
              ) : userStats ? (
                <>
                  <Typography gutterBottom>Total Requests: {userStats.total_requests}</Typography>
                  <Typography gutterBottom>Requests Today: {userStats.requests_today}</Typography>
                  <Typography gutterBottom>Requests This Month: {userStats.requests_this_month}</Typography>
                  <Typography gutterBottom>Rate Limit Remaining: {userStats.rate_limit_remaining}</Typography>
                  <Typography gutterBottom>Rate Limit Resets At: {new Date(userStats.rate_limit_reset).toLocaleString()}</Typography>
                </>
              ) : (
                <Typography>No stats available.</Typography>
              )}
            </Paper>
          </LazyComponent>
        </Grid>
        <Grid item xs={12}>
          <Paper sx={{ p: 2, height: 400 }}>
            <Typography variant="h6" gutterBottom>Analysis</Typography>
            <LazyComponent height="calc(100% - 40px)"> 
              <Typography variant="body1">Analysis data will be displayed here.</Typography>
            </LazyComponent>
          </Paper>
        </Grid>
      </Grid>

      {/* Display Real-time Pricing Data */}
      <Grid item xs={12} mt={3}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>Live Pricing Data</Typography>
            {Object.keys(realtimePrices).length > 0 ? (
              <Box sx={{ maxHeight: '300px', overflowY: 'auto', pr: 2 }}>
                {Object.values(realtimePrices).map((priceData, index) => (
                  <Box key={index} sx={{ mb: 2, p: 1, borderBottom: '1px solid #eee', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Box>
                      <Typography variant="body1" sx={{ fontWeight: 'bold' }}>{priceData.instrument.id}</Typography>
                      <Typography variant="body2"><strong>Spot:</strong> ${priceData.instrument.params.spot.toFixed(2)} | <strong>Strike:</strong> ${priceData.instrument.params.strike.toFixed(2)} | <strong>Vol:</strong> {(priceData.instrument.params.volatility * 100).toFixed(2)}%</Typography>
                    </Box>
                    <Box sx={{ textAlign: 'right' }}>
                      <Typography variant="h6">
                        ${priceData.price.toFixed(4)}
                      </Typography>
                      <Typography variant="body2">Delta: {priceData.greeks.delta.toFixed(4)}</Typography>
                      <Typography variant="body2">Calc Time: {priceData.computation_time_ms.toFixed(2)} ms</Typography>
                      <Typography variant="caption">Updated: {new Date(priceData.calculated_at).toLocaleTimeString()}</Typography>
                    </Box>
                  </Box>
                ))}
              </Box>
            ) : (
              <Typography>Waiting for real-time price updates...</Typography>
            )}
          </Paper>
      </Grid>

      {/* Button to test sending messages */}
      <Box sx={{ mt: 3, textAlign: 'center' }}>
        <Button variant="contained" onClick={handleSendMessage} disabled={!isConnected}>
          Send Test WebSocket Message
        </Button>
        {isConnected ? (
          <Typography variant="caption" color="success.main" sx={{ ml: 2 }}>WebSocket Connected</Typography>
        ) : (
          <Typography variant="caption" color="error.main" sx={{ ml: 2 }}>WebSocket Disconnected</Typography>
        )}
      </Box>
      
      {/* Display latest processed message received */}
      {latestWsMessage && (
        <Box sx={{ mt: 3, p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
          <Typography variant="subtitle1">Latest Processed WebSocket Message:</Typography>
          <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', maxHeight: '100px', overflowY: 'auto', backgroundColor: '#f0f0f0', padding: '8px', borderRadius: '4px' }}>
            {JSON.stringify(latestWsMessage, null, 2)}
          </pre>
        </Box>
      )}
    </Container>
  );
};

export default Dashboard;
