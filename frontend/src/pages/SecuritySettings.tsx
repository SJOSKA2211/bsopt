import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Container, Typography, Button, Box, Paper, Grid, ButtonGroup, Alert, CircularProgress } from '@mui/material';
import apiClient from '@/utils/apiClient'; // Import the API client
import { AxiosError } from 'axios'; 

interface ErrorResponse {
  message?: string;
  detail?: string | unknown;
}

// Import MFA components and types

// Assume UserProfile interface is available or define it here based on backend schema
interface UserProfile {
  id: string;
  email: string;
  full_name: string;
  tier: string;
  is_active: boolean;
  is_verified: boolean;
  is_mfa_enabled: boolean; // Flag indicating MFA status
  created_at: string;
  last_login: string | null;
}

const SecuritySettings: React.FC = () => {
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  const navigate = useNavigate();

  // Fetch user profile to display current security status, including MFA
  useEffect(() => {
    const fetchUserProfile = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await apiClient.get('/users/me');
        setUserProfile(response.data);
      } catch (err: unknown) { // Changed from AxiosError to unknown
        console.error('Failed to fetch user profile:', err);
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
        setError(displayMessage || 'Failed to load security settings.');
      } finally {
        setLoading(false);
      }
    };
    fetchUserProfile();
  }, [navigate]);

  const handleSetupMFA = () => {
    navigate('/mfa/setup');
  };

  const handleDisableMFA = () => {
    navigate('/settings/mfa/disable');
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Security Settings
        </Typography>
        
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '200px' }}>
            <CircularProgress />
          </Box>
        ) : userProfile ? (
          <Grid container spacing={3}>
            {/* MFA Section */}
            <Grid item xs={12}>
              <Box sx={{ mt: 2 }}>
                <Typography variant="h6" gutterBottom>Multi-Factor Authentication (MFA)</Typography>
                
                <Typography variant="body2" sx={{ mb: 2 }}>
                  {userProfile.is_mfa_enabled 
                    ? "MFA is currently enabled for your account. You can disable it below or manage your authenticator app."
                    : "Enhance your account security by enabling Multi-Factor Authentication using an authenticator app."
                  }
                </Typography>

                <ButtonGroup variant="outlined">
                  <Button onClick={handleSetupMFA} disabled={userProfile.is_mfa_enabled}>
                    Setup MFA
                  </Button>
                  <Button onClick={handleDisableMFA} disabled={!userProfile.is_mfa_enabled}>
                    Disable MFA
                  </Button>
                </ButtonGroup>
              </Box>
            </Grid>
            
            {/* Other Security Settings can be added here */}
            {/* e.g., Change Password */}
          </Grid>
        ) : (
          <Typography>Could not load security settings.</Typography>
        )}
      </Paper>
    </Container>
  );
};

export default SecuritySettings;
