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
import { motion } from 'framer-motion';
import {
  Visibility,
  VisibilityOff,
  TrendingUpOutlined as ChartIcon,
  GitHub as GitHubIcon,
  Google as GoogleIcon,
  LockOutlined as LockIcon,
  AlternateEmailOutlined as MailIcon,
} from '@mui/icons-material';
import { authClient } from '../../lib/auth-client';

// Decorative Greek / finance symbols for background
const GREEK_SYMBOLS = ['Δ', 'Γ', 'Θ', 'Ρ', 'Σ', 'Λ', 'Φ', 'Ψ', '∑', '∂'];

const DecorativeBg: React.FC = () => (
  <>
    {GREEK_SYMBOLS.map((sym, i) => (
      <Typography
        key={i}
        className="animate-float"
        sx={{
          position: 'absolute',
          color: alpha('#10b981', 0.04 + (i % 3) * 0.015),
          fontSize: `${48 + (i % 4) * 24}px`,
          fontWeight: 700,
          top: `${(i * 13 + 5) % 85}%`,
          left: `${(i * 17 + 3) % 90}%`,
          userSelect: 'none',
          pointerEvents: 'none',
          fontFamily: 'serif',
          filter: 'blur(0.5px)',
          animationDelay: `${i * 0.5}s`,
        }}
      >
        {sym}
      </Typography>
    ))}
    {/* Background glow orbs */}
    <Box sx={{ position: 'absolute', top: '8%', right: '10%', width: 400, height: 400, borderRadius: '50%', bgcolor: alpha('#10b981', 0.07), filter: 'blur(80px)', pointerEvents: 'none' }} />
    <Box sx={{ position: 'absolute', bottom: '12%', left: '5%', width: 350, height: 350, borderRadius: '50%', bgcolor: alpha('#38bdf8', 0.07), filter: 'blur(70px)', pointerEvents: 'none' }} />
    <Box sx={{ position: 'absolute', top: '50%', left: '42%', width: 280, height: 280, borderRadius: '50%', bgcolor: alpha('#a855f7', 0.05), filter: 'blur(60px)', pointerEvents: 'none' }} />
    {/* Subtle grid lines */}
    <Box
      sx={{
        position: 'absolute',
        inset: 0,
        backgroundImage: `
          linear-gradient(rgba(148,163,184,0.03) 1px, transparent 1px),
          linear-gradient(90deg, rgba(148,163,184,0.03) 1px, transparent 1px)
        `,
        backgroundSize: '48px 48px',
        pointerEvents: 'none',
      }}
    />
  </>
);

export default function SignIn() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);

  const signIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess(false);

    await authClient.signIn.email({ email, password }, {
      onRequest: () => setLoading(true),
      onSuccess: () => { setLoading(false); setSuccess(true); },
      onError: (ctx: any) => { setLoading(false); setError(ctx.error.message); },
    });
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        bgcolor: 'background.default',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <DecorativeBg />

      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        style={{ width: '100%', maxWidth: 440, zIndex: 1 }}
      >
        <Paper
          className="qfd-glass qfd-holographic"
          sx={{
            position: 'relative',
            width: '100%',
            p: 4,
            border: `1px solid ${alpha('#94a3b8', 0.1)}`,
            boxShadow: `0 32px 80px rgba(0,0,0,0.5), 0 0 0 1px ${alpha('#10b981', 0.08)}`,
          }}
        >
        {/* Brand */}
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
          Welcome back
        </Typography>
        <Typography variant="body2" sx={{ color: 'text.disabled', mb: 2.5, fontSize: '0.82rem' }}>
          Sign in to access your trading dashboard
        </Typography>

        <Box component="form" onSubmit={signIn}>
          {error && <Alert severity="error" sx={{ mb: 2, borderRadius: 2 }}>{error}</Alert>}
          {success && <Alert severity="success" sx={{ mb: 2, borderRadius: 2 }}>Signed in successfully!</Alert>}

          <TextField
            margin="normal"
            required
            fullWidth
            id="email"
            label="Email Address"
            type="email"
            autoComplete="email"
            autoFocus
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
            sx={{ mt: 0 }}
          />

          <TextField
            margin="normal"
            required
            fullWidth
            id="password"
            label="Password"
            type={showPassword ? 'text' : 'password'}
            autoComplete="current-password"
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

          <Stack direction="row" justifyContent="flex-end" sx={{ mt: 0.5, mb: 2.5 }}>
            <Link href="#" variant="caption" sx={{ color: 'text.disabled', textDecoration: 'none', '&:hover': { color: 'primary.main' } }}>
              Forgot password?
            </Link>
          </Stack>

          <Button
            type="submit"
            fullWidth
            variant="contained"
            size="large"
            disabled={loading}
            sx={{ py: 1.5, fontSize: '0.95rem', mb: 2 }}
          >
            {loading ? (
              <CircularProgress size={24} color="inherit" aria-label="Signing in..." />
            ) : (
              'Sign In'
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
          Don&apos;t have an account?{' '}
          <Link href="/signup" sx={{ color: 'primary.main', textDecoration: 'none' }}>
            Sign Up
          </Link>
        </Typography>

        <Stack direction="row" spacing={2.5} justifyContent="center" sx={{ mt: 2.5 }}>
          {['Privacy', 'Terms', 'Support'].map((link) => (
            <Link key={link} href="#" variant="caption" sx={{ color: 'text.disabled', textDecoration: 'none', fontSize: '0.68rem', '&:hover': { color: 'text.secondary' } }}>
              {link}
            </Link>
          ))}
        </Stack>
        </Paper>
      </motion.div>
    </Box>
  );
}
