import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Container, Typography, TextField, Button, Box, Paper, Alert, CircularProgress } from '@mui/material';
import apiClient from '@/utils/apiClient'; // Import the API client
import { AxiosError } from 'axios'; 

interface ErrorResponse {
  message?: string;
  detail?: string | unknown;
}

const MFADisable: React.FC = () => {
  const [mfaCode, setMfaCode] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  
  const navigate = useNavigate();

  const handleDisableMFA = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccessMessage(null);

    try {
      // Call the backend endpoint to disable MFA
      // The backend is expected to verify the MFA code provided
      const response = await apiClient.post('/auth/mfa/disable', { code: mfaCode });

      if (response.data && response.data.message) {
        setSuccessMessage(response.data.message);
        // After disabling MFA, it's good practice to re-authenticate or inform the user.
        // For simplicity, we'll redirect to the login page after a short delay.
        setTimeout(() => {
          // Clear auth tokens and redirect to login
          localStorage.removeItem('accessToken');
          localStorage.removeItem('refreshToken');
          navigate('/login'); 
        }, 3000); // Redirect after 3 seconds
      }
    } catch (err: unknown) { // Changed from AxiosError to unknown
      console.error('MFA disable failed:', err);
      let displayMessage: string | null = null;
      if (typeof err === 'object' && err !== null && 'response' in err) {
        const axiosError = err as AxiosError;
        const errorData = axiosError.response?.data as ErrorResponse;
        if (typeof errorData.message === 'string') {
          displayMessage = errorData.message;
        } else if (typeof errorData.detail === 'string') {
          displayMessage = errorData.detail;
        }
      }
      setError(displayMessage || 'An unexpected error occurred. Please try again later.');
      // Clear MFA code input on error to allow re-entry
      setMfaCode('');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="sm" sx={{ mt: 8 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Disable Multi-Factor Authentication
        </Typography>
        <Typography variant="body1" gutterBottom align="center" sx={{ mb: 3 }}>
          Enter your current MFA code to disable this security feature.
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        {successMessage && (
          <Alert severity="success" sx={{ mb: 2 }}>
            {successMessage}
          </Alert>
        )}
        
        <Box component="form" onSubmit={handleDisableMFA} noValidate sx={{ mt: 1 }}>
          <TextField
            margin="normal"
            required
            fullWidth
            id="mfaCode"
            label="Enter 6-Digit MFA Code"
            name="mfaCode"
            autoComplete="one-time-code"
            value={mfaCode}
            onChange={(e) => setMfaCode(e.target.value)}
            sx={{ mb: 2 }}
            disabled={loading || !!successMessage} // Disable input when loading or on success
          />
          <Button
            type="submit"
            fullWidth
            variant="contained"
            color="error" // Use error color for disabling a security feature
            sx={{ mt: 3, mb: 2 }}
            disabled={loading || !!successMessage}
          >
            {loading ? <CircularProgress size={24} sx={{ color: 'white' }} /> : 'Disable MFA'}
          </Button>
        </Box>
        
        <Box sx={{ mt: 2, textAlign: 'center' }}>
          <Typography variant="body2">
            Need help? <Button onClick={() => navigate('/help')}>Contact Support</Button>
          </Typography>
        </Box>
      </Paper>
    </Container>
  );
};

export default MFADisable;
