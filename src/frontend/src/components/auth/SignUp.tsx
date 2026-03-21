import { useState } from 'react';
import {
  Box,
  Button,
  TextField,
  Typography,
  Paper,
  Alert,
  Stack,
  Divider,
  alpha,
  InputAdornment,
  IconButton,
  Link,
  CircularProgress,
} from '@mui/material';
import {
  Visibility,
  VisibilityOff,
  TrendingUpOutlined as ChartIcon,
  GitHub as GitHubIcon,
  Google as GoogleIcon,
  LockOutlined as LockIcon,
  AlternateEmailOutlined as MailIcon,
  PersonOutline as PersonIcon,
} from '@mui/icons-material';
import { authClient } from '../../lib/auth-client';
import { useNavigate } from 'react-router-dom';

export function SignUp() {
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const navigate = useNavigate();

  const signUp = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess(false);

    await authClient.signUp.email({ 
      email, 
      password, 
      name 
    }, {
      onRequest: () => setLoading(true),
      onSuccess: () => { 
        setLoading(false); 
        setSuccess(true);
        setTimeout(() => navigate('/login'), 2000);
      },
      onError: (ctx: any) => { 
        setLoading(false); 
        setError(ctx.error.message); 
      },
    });
  };

  return (
    <Paper
      className="fade-in"
      sx={{
        position: 'relative',
        zIndex: 1,
        width: '100%',
        maxWidth: 440,
        mx: 'auto',
        p: 4,
        border: `1px solid ${alpha('#94a3b8', 0.1)}`,
        boxShadow: `0 32px 80px rgba(0,0,0,0.5), 0 0 0 1px ${alpha('#10b981', 0.08)}`,
      }}
    >
      <Stack direction="row" spacing={1.5} alignItems="center" justifyContent="center" sx={{ mb: 1.5 }}>
        <Box
          sx={{
            width: 44,
            height: 44,
            borderRadius: 2.5,
            background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: `0 8px 20px ${alpha('#10b981', 0.4)}`,
          }}
        >
          <ChartIcon sx={{ color: '#fff', fontSize: 24 }} />
        </Box>
        <Box>
          <Typography
            variant="h5"
            sx={{
              fontWeight: 800,
              background: 'linear-gradient(135deg, #10b981, #38bdf8)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              lineHeight: 1.1,
            }}
          >
            BS-Opt Pro
          </Typography>
          <Typography variant="caption" sx={{ color: 'text.disabled', fontSize: '0.65rem', letterSpacing: '0.06em' }}>
            QUANTITATIVE OPTIONS TERMINAL
          </Typography>
        </Box>
      </Stack>

      <Typography
        variant="body2"
        sx={{ textAlign: 'center', color: 'text.disabled', mb: 3.5, mt: 0.5, fontSize: '0.82rem' }}
      >
        Access institutional-grade volatility surfaces &amp; real-time Greeks
      </Typography>

      <Typography variant="h6" sx={{ fontWeight: 700, mb: 0.5 }}>
        Create an account
      </Typography>
      <Typography variant="body2" sx={{ color: 'text.disabled', mb: 2.5, fontSize: '0.82rem' }}>
        Join the BS-Opt professional network
      </Typography>

      <Box component="form" onSubmit={signUp}>
        {error && <Alert severity="error" sx={{ mb: 2, borderRadius: 2 }}>{error}</Alert>}
        {success && <Alert severity="success" sx={{ mb: 2, borderRadius: 2 }}>Account created! Redirecting to login...</Alert>}

        <TextField
          margin="normal"
          required
          fullWidth
          id="name"
          label="Full Name"
          name="name"
          autoComplete="name"
          autoFocus
          value={name}
          onChange={(e) => setName(e.target.value)}
          disabled={loading}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <PersonIcon sx={{ fontSize: 18, color: 'text.disabled' }} />
              </InputAdornment>
            ),
          }}
          sx={{ mt: 0 }}
        />

        <TextField
          margin="normal"
          required
          fullWidth
          id="email"
          label="Email Address"
          name="email"
          type="email"
          autoComplete="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          disabled={loading}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <MailIcon sx={{ fontSize: 18, color: 'text.disabled' }} />
              </InputAdornment>
            ),
          }}
        />

        <TextField
          margin="normal"
          required
          fullWidth
          id="password"
          label="Password"
          name="password"
          type={showPassword ? 'text' : 'password'}
          autoComplete="new-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          disabled={loading}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <LockIcon sx={{ fontSize: 18, color: 'text.disabled' }} />
              </InputAdornment>
            ),
            endAdornment: (
              <InputAdornment position="end">
                <IconButton
                  size="small"
                  onClick={() => setShowPassword((s) => !s)}
                  edge="end"
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? <VisibilityOff fontSize="small" /> : <Visibility fontSize="small" />}
                </IconButton>
              </InputAdornment>
            ),
          }}
        />

        <Button
          type="submit"
          fullWidth
          variant="contained"
          size="large"
          disabled={loading}
          sx={{ py: 1.5, fontSize: '0.95rem', mt: 2, mb: 2 }}
        >
          {loading ? (
            <CircularProgress size={24} color="inherit" aria-label="Signing up" />
          ) : (
            'Sign Up'
          )}
        </Button>
      </Box>

      <Divider sx={{ my: 2.5 }}>
        <Typography variant="caption" sx={{ color: 'text.disabled', px: 1 }}>
          or continue with
        </Typography>
      </Divider>

      <Stack direction="row" spacing={1.5}>
        <Button
          fullWidth
          variant="outlined"
          startIcon={<GoogleIcon />}
          sx={{ py: 1, fontSize: '0.82rem', borderColor: alpha('#94a3b8', 0.15), color: 'text.secondary', '&:hover': { borderColor: alpha('#94a3b8', 0.3) } }}
        >
          Google
        </Button>
        <Button
          fullWidth
          variant="outlined"
          startIcon={<GitHubIcon />}
          sx={{ py: 1, fontSize: '0.82rem', borderColor: alpha('#94a3b8', 0.15), color: 'text.secondary', '&:hover': { borderColor: alpha('#94a3b8', 0.3) } }}
        >
          GitHub
        </Button>
      </Stack>

      <Typography variant="caption" sx={{ display: 'block', textAlign: 'center', color: 'text.disabled', mt: 3 }}>
        Already have an account?{' '}
        <Link href="/login" sx={{ color: 'primary.main', textDecoration: 'none' }}>
          Sign In
        </Link>
      </Typography>
    </Paper>
  );
}
