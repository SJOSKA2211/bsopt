import { useState } from 'react';
import {
  Box,
  Button,
  TextField,
  Typography,
  Alert,
  Stack,
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
  <div className="flex items-center gap-3 mb-6">
    <div className="relative w-2 h-2">
      <div className="status-pill healthy w-2 h-2 p-0 rounded-full bg-mint" />
      <div className="absolute -inset-1 rounded-full border border-mint/30 animate-pulse" />
    </div>
    <span className="text-[10px] text-mint font-bold tracking-[0.1em] uppercase">
      Manifold_L1_Active
    </span>
  </div>
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
    <div className="min-h-screen bg-bento-bg flex items-center justify-center p-4">
      <AnimatedCard
        className="w-full max-w-[420px] !p-10"
      >
        <SystemStatus />

        <div className="flex flex-col gap-1 mb-8">
          <div className="flex items-center gap-3">
            <ChartIcon className="text-mint text-2xl" />
            <h1 className="text-2xl font-extrabold tracking-tight text-white">
              BSOPT_PRO
            </h1>
          </div>
          <p className="text-sm text-white/60 font-medium">
            Institutional Quantitative Analytics
          </p>
        </div>

        <form onSubmit={signIn} className="space-y-6">
          {error && <Alert severity="error" className="rounded-xl">{error}</Alert>}

          <div className="flex flex-col gap-4">
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
                    <MailIcon className="text-white/40" sx={{ fontSize: 20 }} />
                  </InputAdornment>
                ),
                className: "bg-white/5 rounded-xl border border-white/10 focus-within:border-mint transition-colors"
                ,
                sx: { 
                    '& .MuiFilledInput-input': { color: '#fff' }
                }
              }}
              InputLabelProps={{ className: "!text-white/40" }}
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
                    <LockIcon className="text-white/40" sx={{ fontSize: 20 }} />
                  </InputAdornment>
                ),
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton size="small" onClick={() => setShowPassword(!showPassword)} className="text-white/40">
                      {showPassword ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  </InputAdornment>
                ),
                className: "bg-white/5 rounded-xl border border-white/10 focus-within:border-mint transition-colors",
                sx: { 
                    '& .MuiFilledInput-input': { color: '#fff' }
                }
              }}
              InputLabelProps={{ className: "!text-white/40" }}
            />

            <Button
              type="submit"
              fullWidth
              disabled={loading}
              className="!py-4 !rounded-xl !bg-mint !text-black font-bold text-sm tracking-wide hover:!bg-teal transition-all disabled:opacity-50"
            >
              {loading ? <CircularProgress size={24} className="text-black" /> : 'Initialize Terminal Access'}
            </Button>
          </div>
        </form>

        <div className="flex items-center justify-center gap-2 mt-8">
          <span className="text-xs text-white/40">New Operative?</span>
          <Link href="/signup" className="text-xs text-mint font-bold no-underline hover:text-teal">
            Request Access
          </Link>
        </div>

        <div className="flex justify-center gap-6 mt-6 opacity-30">
          <GoogleIcon className="text-lg cursor-pointer hover:opacity-100 transition-opacity" />
          <GitHubIcon className="text-lg cursor-pointer hover:opacity-100 transition-opacity" />
        </div>
      </AnimatedCard>
    </div>
  );
}
