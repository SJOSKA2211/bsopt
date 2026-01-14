import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Container, Typography, TextField, Button, Box, Paper, Alert } from '@mui/material';
import apiClient from '@/utils/apiClient'; // Import the API client
import { AxiosError } from 'axios'; 

interface ErrorResponse {
  message?: string;
  detail?: string | unknown;
}

const ResetPassword: React.FC = () => {
  const { token } = useParams<{ token: string }>(); // Get token from URL parameters
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (!token) {
      setError('Invalid or missing reset token.');
      // Optionally redirect to forgot password page if token is missing
      // navigate('/forgot-password');
    }
  }, [token, navigate]);

  const handleResetPassword = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccessMessage(null);

    if (!token) {
      setError('Invalid request. Missing reset token.');
      return;
    }

    if (newPassword !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    try {
      const response = await apiClient.post('/auth/password-reset/confirm', {
        token: token,
        new_password: newPassword,
      });

      if (response.data && response.data.message) {
        setSuccessMessage(response.data.message);
        // Redirect to login page after successful password reset
        setTimeout(() => navigate('/login'), 5000); // Redirect after 5 seconds
      }
    } catch (err: unknown) { // Changed from AxiosError to unknown
      console.error('Password reset confirmation failed:', err);
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
      setError(displayMessage || 'Password reset failed. Please try again.');
    }
  };

  return (
    <Container maxWidth="sm" sx={{ mt: 8 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Reset Password
        </Typography>
        
        {!token && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            Invalid or missing reset token. Please request a new reset link.
          </Alert>
        )}

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
        
        <Box component="form" onSubmit={handleResetPassword} noValidate sx={{ mt: 1 }}>
          <TextField
            margin="normal"
            required
            fullWidth
            id="newPassword"
            label="New Password"
            name="newPassword"
            type="password"
            autoComplete="new-password"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            sx={{ mb: 2 }}
            disabled={!token || !!successMessage}
          />
          <TextField
            margin="normal"
            required
            fullWidth
            id="confirmPassword"
            label="Confirm New Password"
            name="confirmPassword"
            type="password"
            autoComplete="new-password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            sx={{ mb: 2 }}
            disabled={!token || !!successMessage}
          />
          <Button
            type="submit"
            fullWidth
            variant="contained"
            color="primary"
            sx={{ mt: 3, mb: 2 }}
            disabled={!token || !!successMessage}
          >
            Reset Password
          </Button>
        </Box>
      </Paper>
    </Container>
  );
};

export default ResetPassword;