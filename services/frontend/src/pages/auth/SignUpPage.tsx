import { SignUp } from '../../components/auth/SignUp';
import { Box, Typography, alpha } from '@mui/material';

// Decorative Greek / finance symbols for background
const GREEK_SYMBOLS = ['Δ', 'Γ', 'Θ', 'Ρ', 'Σ', 'Λ', 'Φ', 'Ψ', '∑', '∂'];

const DecorativeBg: React.FC = () => (
  <>
    {GREEK_SYMBOLS.map((sym, i) => (
      <Typography
        key={i}
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

export default function SignUpPage() {
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
      <Box sx={{ position: 'relative', zIndex: 1, width: '100%' }}>
        <SignUp />
      </Box>
    </Box>
  );
}
