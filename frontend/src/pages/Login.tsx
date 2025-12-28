import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Container, Typography, TextField, Button, Box, Paper, Alert, CircularProgress } from '@mui/material';
import apiClient from '@/utils/apiClient'; // Import the API client
import { AxiosError } from 'axios'; 

interface ErrorResponse {
  message?: string;
  detail?: string | unknown;
}

// Interface for the login response, including the MFA flag
interface LoginResponse {
  access_token: string;
  refresh_token: string;
  expires_in: number;
  user_id: string;
  email: string;
  tier: string;
  requires_mfa: boolean; // Flag indicating if MFA is needed for this login
}

const Login: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [mfaCode, setMfaCode] = useState(''); // State for MFA code input
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [requiresMFA, setRequiresMFA] = useState<boolean>(false); // State to track if MFA is required

  const navigate = useNavigate();
  const location = useLocation();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null); // Clear previous errors
    setLoading(true);

    try {
      const response = await apiClient.post<LoginResponse>('/auth/login', {
        email: email,
        password: password,
        mfa_code: requiresMFA ? mfaCode : undefined, // Only send MFA code if required
      });

      const { access_token, refresh_token, user_id, tier, requires_mfa: backendRequiresMFA } = response.data;

      // Handle MFA requirement
      if (backendRequiresMFA && !requiresMFA) {
        // Backend requires MFA, but we haven't been prompted yet.
        // Show the MFA input field and update state.
        setRequiresMFA(true);
        setError(null); // Clear any previous login errors
        setLoading(false); // Stop loading for login attempt, wait for MFA code
        return; 
      }
      
      // Successful login (either with or without MFA, or after MFA verification)
      localStorage.setItem('accessToken', access_token);
      localStorage.setItem('refreshToken', refresh_token);
      localStorage.setItem('userId', user_id);
      localStorage.setItem('userTier', tier); 

      // Redirect to the intended page or dashboard
      const { from } = location.state || { from: { pathname: "/dashboard" } };
      navigate(from.pathname || "/dashboard", { replace: true });

    } catch (err: unknown) { // Changed from AxiosError to unknown
      console.error('Login failed:', err);
      let displayMessage: string | null = null;
      if (typeof err === 'object' && err !== null && 'response' in err) {
        const axiosError = err as AxiosError;
        const errorData = axiosError.response?.data as ErrorResponse;
        if (typeof errorData.message === 'string') {
          displayMessage = errorData.message;
        } else if (typeof errorData.detail === 'string') {
          displayMessage = errorData.detail;
        }
        
        // Specific check for MFA code error
        if (axiosError.response?.status === 401 && (errorData?.detail as string) === "Invalid MFA code") { // Cast detail to string for comparison
          setError("Invalid MFA code. Please try again.");
          setRequiresMFA(true); // Ensure MFA step remains visible
        } else if (axiosError.response?.status === 403 && (typeof errorData?.detail === 'string' && errorData.detail.includes("not verified"))) {
          setError("Email not verified. Please check your email for verification link.");
        } else {
           setError(displayMessage || 'Login failed. Please check your credentials.');
        }
      } else if (typeof err === 'object' && err !== null && 'request' in err) {
        setError('Network error. Could not connect to the server.');
      } else {
        setError('An unexpected error occurred. Please try again later.');
      }
      // If login fails, reset MFA requirement state
      setRequiresMFA(false); 
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="sm" sx={{ mt: 8 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Login
        </Typography>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        <Box component="form" onSubmit={handleLogin} noValidate sx={{ mt: 1 }}>
          <TextField
            margin="normal"
            required
            fullWidth
            id="email"
            label="Email Address"
            name="email"
            autoComplete="email"
            autoFocus
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            sx={{ mb: 2 }}
            disabled={loading} // Disable input while loading
          />
          <TextField
            margin="normal"
            required
            fullWidth
            id="password"
            label="Password"
            name="password"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            sx={{ mb: 2 }}
            disabled={loading} // Disable input while loading
          />

          {/* MFA Code Input - conditionally rendered */}
          {requiresMFA && (
            <TextField
              margin="normal"
              required
              fullWidth
              id="mfaCode"
              label="MFA Code"
              name="mfaCode"
              autoComplete="one-time-code"
              value={mfaCode}
              onChange={(e) => setMfaCode(e.target.value)}
              sx={{ mb: 2 }}
              disabled={loading}
            />
          )}

          <Button
            type="submit"
            fullWidth
            variant="contained"
            color="primary"
            sx={{ mt: 3, mb: 2 }}
            disabled={loading}
          >
            {loading ? <CircularProgress size={24} sx={{ color: 'white' }} /> : requiresMFA ? 'Verify MFA' : 'Sign In'}
          </Button>
        </Box>
        
        <Box sx={{ mt: 2, textAlign: 'center' }}>
          <Typography variant="body2">
            Don't have an account? <Button onClick={() => navigate('/register')}>Sign Up</Button>
          </Typography>
          <Typography variant="body2" sx={{ mt: 1 }}>
            <Button onClick={() => navigate('/forgot-password')}>Forgot Password?</Button>
          </Typography>
        </Box>
      </Paper>
    </Container>
  );
};

export default Login;