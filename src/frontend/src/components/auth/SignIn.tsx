import { useState } from 'react';
import {
  Box,
  Button,
  TextField,
  Typography,
  Alert,
  Stack,
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
} from '@mui/icons-material';
import { AnimatedCard } from '../common/AnimatedCard';

const SystemStatus: React.FC = () => (
  <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 3 }}>
    <Box sx={{ position: 'relative', width: 8, height: 8 }}>
      <Box className="status-pill healthy" sx={{ width: 8, height: 8, p: 0, borderRadius: '50%', background: 'var(--accent-mint)' }} />
      <Box sx={{ position: 'absolute', inset: -4, borderRadius: '50%', border: '1px solid var(--accent-mint)', opacity: 0.3 }} className="shimmer-overlay" />
    </Box>
    <Typography variant="caption" sx={{ color: 'var(--accent-mint)', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
      Manifold_L1_Active
    </Typography>
  </Stack>
);

export default function SignIn() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const signIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    // Mock authentication logic
    setTimeout(() => {
      setLoading(false);
      window.location.href = '/';
    }, 1500);
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        background: 'var(--bento-bg)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
        p: 2,
      }}
    >
      <AnimatedCard
        className="glass-panel"
        sx={{ width: '100%', maxWidth: 420, p: '40px !important' }}
      >
        <SystemStatus />

        <Stack spacing={0.5} sx={{ mb: 4 }}>
          <Stack direction="row" spacing={1.5} alignItems="center">
            <ChartIcon sx={{ color: 'var(--accent-mint)', fontSize: 24 }} />
            <Typography variant="h5" sx={{ fontWeight: 800, letterSpacing: '-0.02em' }}>
              BSOPT_PRO
            </Typography>
          </Stack>
          <Typography variant="body2" sx={{ color: 'var(--text-secondary)' }}>
            Institutional Quantitative Analytics
          </Typography>
        </Stack>

        <Box component="form" onSubmit={signIn}>
          {error && <Alert severity="error" sx={{ mb: 2, borderRadius: '12px' }}>{error}</Alert>}

          <Stack spacing={2}>
            <TextField
              fullWidth
              label="Quant ID (Email)"
              type="email"
              variant="filled"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              InputProps={{
                disableUnderline: true,
                startAdornment: (
                  <InputAdornment position="start">
                    <MailIcon sx={{ fontSize: 20, color: 'var(--text-secondary)' }} />
                  </InputAdornment>
                ),
                sx: { 
                  background: 'rgba(255,255,255,0.03)', 
                  borderRadius: '12px',
                  border: '1px solid rgba(255,255,255,0.05)',
                  '&:focus-within': { border: '1px solid var(--accent-mint)' }
                }
              }}
              InputLabelProps={{ sx: { color: 'var(--text-secondary)' } }}
            />

            <TextField
              fullWidth
              label="Secure Key"
              type={showPassword ? 'text' : 'password'}
              variant="filled"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              InputProps={{
                disableUnderline: true,
                startAdornment: (
                  <InputAdornment position="start">
                    <LockIcon sx={{ fontSize: 20, color: 'var(--text-secondary)' }} />
                  </InputAdornment>
                ),
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton size="small" onClick={() => setShowPassword(!showPassword)} sx={{ color: 'var(--text-secondary)' }}>
                      {showPassword ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  </InputAdornment>
                ),
                sx: { 
                  background: 'rgba(255,255,255,0.03)', 
                  borderRadius: '12px',
                  border: '1px solid rgba(255,255,255,0.05)',
                  '&:focus-within': { border: '1px solid var(--accent-mint)' }
                }
              }}
              InputLabelProps={{ sx: { color: 'var(--text-secondary)' } }}
            />

            <Button
              type="submit"
              fullWidth
              disabled={loading}
              sx={{ 
                py: 1.8, 
                borderRadius: '12px', 
                background: 'var(--accent-mint)', 
                color: '#000',
                fontWeight: 700,
                fontSize: '0.9rem',
                '&:hover': { background: 'var(--accent-teal)' }
              }}
            >
              {loading ? <CircularProgress size={24} sx={{ color: '#000' }} /> : 'Initialize Terminal Access'}
            </Button>
          </Stack>
        </Box>

        <Stack direction="row" spacing={1} justifyContent="center" sx={{ mt: 4 }}>
          <Typography variant="caption" sx={{ color: 'var(--text-secondary)' }}>
            New Operative?
          </Typography>
          <Link href="/signup" sx={{ color: 'var(--accent-mint)', fontWeight: 600, textDecoration: 'none', fontSize: '0.75rem' }}>
            Request Access
          </Link>
        </Stack>

        <Stack direction="row" spacing={3} justifyContent="center" sx={{ mt: 3, opacity: 0.5 }}>
          <GoogleIcon sx={{ fontSize: 18, cursor: 'pointer' }} />
          <GitHubIcon sx={{ fontSize: 18, cursor: 'pointer' }} />
        </Stack>
      </AnimatedCard>
    </Box>
  );
}
