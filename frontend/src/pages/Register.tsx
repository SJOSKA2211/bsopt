import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Container, Typography, TextField, Button, Box, Paper, Alert, CircularProgress, FormControlLabel, Checkbox } from '@mui/material';
import apiClient from '@/utils/apiClient'; // Import the API client
import { AxiosError } from 'axios'; 

interface ErrorResponse {
  message?: string;
  detail?: string | unknown;
}

// Assuming RegisterRequest and RegisterResponse are defined elsewhere or can be inferred
interface RegisterRequest { // eslint-disable-line @typescript-eslint/no-unused-vars 
  full_name: string;
  email: string;
  password?: string; 
  password_confirm?: string;
  accept_terms: boolean;
}

interface RegisterResponse {
  user_id: string;
  email: string;
  message: string;
  verification_required: boolean;
  requires_mfa_setup: boolean; // Flag indicating if MFA setup is recommended/required next
}

const Register: React.FC = () => {
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [acceptTerms, setAcceptTerms] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  
  const navigate = useNavigate();

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccessMessage(null);

    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    if (!acceptTerms) {
      setError('You must accept the terms and conditions.');
      return;
    }

    setLoading(true);
    try {
      const response = await apiClient.post<RegisterResponse>('/auth/register', {
        full_name: fullName,
        email: email,
        password: password,
        password_confirm: confirmPassword,
        accept_terms: acceptTerms
      });

      if (response.data && response.data.message) {
        setSuccessMessage(response.data.message);
        
        // If backend indicates MFA setup is recommended/required next
        if (response.data.requires_mfa_setup) {
          // Redirect to MFA setup page after a short delay
          setTimeout(() => navigate('/mfa/setup'), 5000); 
        } else {
          // Otherwise, redirect to login after a delay
          setTimeout(() => navigate('/login'), 5000); 
        }
      }
    } catch (err: unknown) { // Changed from AxiosError to unknown
      console.error('Registration failed:', err);
      let displayMessage: string | null = null;
      if (typeof err === 'object' && err !== null && 'response' in err) {
        const axiosError = err as AxiosError;
        const errorData = axiosError.response?.data as ErrorResponse;
        
        if (errorData?.detail && Array.isArray(errorData.detail)) {
            // Handle Pydantic validation errors
            displayMessage = errorData.detail.map((err: { msg: string }) => err.msg).join('; ');
        } else if (typeof errorData?.message === 'string') {
          displayMessage = errorData.message;
        } else if (typeof errorData?.detail === 'string') {
          displayMessage = errorData.detail;
        }
      }
      setError(displayMessage || 'Registration failed. Please check your input.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="sm" sx={{ mt: 8 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Register
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
        <Box component="form" onSubmit={handleRegister} noValidate sx={{ mt: 1 }}>
          <TextField
            margin="normal"
            required
            fullWidth
            id="fullName"
            label="Full Name"
            name="fullName"
            autoComplete="name"
            autoFocus
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
            sx={{ mb: 2 }}
            disabled={loading || !!successMessage}
          />
          <TextField
            margin="normal"
            required
            fullWidth
            id="email"
            label="Email Address"
            name="email"
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            sx={{ mb: 2 }}
            disabled={loading || !!successMessage}
          />
          <TextField
            margin="normal"
            required
            fullWidth
            id="password"
            label="Password"
            name="password"
            type="password"
            autoComplete="new-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            sx={{ mb: 2 }}
            disabled={loading || !!successMessage}
          />
          <TextField
            margin="normal"
            required
            fullWidth
            id="confirmPassword"
            label="Confirm Password"
            name="confirmPassword"
            type="password"
            autoComplete="new-password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            sx={{ mb: 2 }}
            disabled={loading || !!successMessage}
          />
          <FormControlLabel
            control={<Checkbox checked={acceptTerms} onChange={(e) => setAcceptTerms(e.target.checked)} color="primary" />}
            label="I accept the terms and conditions"
            disabled={loading || !!successMessage}
          />
          <Button
            type="submit"
            fullWidth
            variant="contained"
            color="primary"
            sx={{ mt: 3, mb: 2 }}
            disabled={loading || !!successMessage}
          >
            {loading ? <CircularProgress size={24} sx={{ color: 'white' }} /> : 'Sign Up'}
          </Button>
        </Box>
        <Box sx={{ mt: 2, textAlign: 'center' }}>
          <Typography variant="body2">
            Already have an account? <Button onClick={() => navigate('/login')}>Sign In</Button>
          </Typography>
        </Box>
      </Paper>
    </Container>
  );
};

export default Register;
