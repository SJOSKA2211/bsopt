import React from 'react';
import {
  Container,
  Typography,
  Paper,
  Box,
  Grid,
  Stack,
  Switch,
  Slider,
  Button,
  Divider,
  alpha,
  useTheme,
  TextField,
  MenuItem,
} from '@mui/material';
import { motion } from 'framer-motion';
import {
  Security,
  Bolt,
  SettingsInputComponent,
  Wifi,
  History,
  ShieldMoon,
  AccountBalanceWallet,
} from '@mui/icons-material';

const SettingsSection: React.FC<{ title: string; icon: React.ReactNode; children: React.ReactNode }> = ({ title, icon, children }) => {
  const theme = useTheme();
  return (
    <Box sx={{ mb: 6 }}>
      <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 3 }}>
        <Box sx={{ color: 'primary.main', display: 'flex' }}>{icon}</Box>
        <Typography variant="h6" sx={{ fontWeight: 900, fontFamily: 'Outfit', letterSpacing: '-0.02em' }}>
          {title}
        </Typography>
      </Stack>
      <Paper
        sx={{
          p: 4,
          borderRadius: 6,
          background: `linear-gradient(135deg, ${alpha('#0f172a', 0.6)}, ${alpha('#0f172a', 0.2)})`,
          backdropFilter: 'blur(40px) saturate(200%)',
          border: `1px solid ${alpha('#fff', 0.05)}`,
        }}
      >
        {children}
      </Paper>
    </Box>
  );
};

const SettingRow: React.FC<{ label: string; description: string; action: React.ReactNode }> = ({ label, description, action }) => (
  <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ py: 2 }}>
    <Box>
      <Typography variant="body1" sx={{ fontWeight: 700 }}>{label}</Typography>
      <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500 }}>{description}</Typography>
    </Box>
    {action}
  </Stack>
);

export const SettingsPage: React.FC = () => {
  const theme = useTheme();
  // @ts-ignore
  const qfd = theme.palette.financial?.qfd;

  return (
    <Container maxWidth="lg" sx={{ mt: 2, pb: 10 }}>
      {/* Header */}
      <Box sx={{ mb: 6 }}>
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Typography
            variant="h3"
            sx={{
              fontWeight: 950,
              fontFamily: 'Outfit',
              letterSpacing: '-0.05em',
              background: `linear-gradient(135deg, ${qfd?.quantum ?? '#00FFFF'}, ${qfd?.nebula ?? '#7B68EE'})`,
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            System Configuration
          </Typography>
          <Typography variant="body1" sx={{ color: 'text.secondary', fontWeight: 600, mt: 1 }}>
            Manage your quantum kernels, API connectivity, and vault security.
          </Typography>
        </motion.div>
      </Box>

      <Grid container spacing={4}>
        <Grid size={{ xs: 12, lg: 8 }}>
          <SettingsSection title="Quantum Preferences" icon={<Bolt />}>
            <SettingRow
              label="Engine Precision"
              description="Adjust the number of qubits for Amplitude Estimation (Higher = More Accurate but Slower)"
              action={
                <Box sx={{ width: 150 }}>
                  <Slider
                    defaultValue={7}
                    step={1}
                    marks
                    min={3}
                    max={12}
                    valueLabelDisplay="auto"
                    sx={{ color: qfd?.quantum }}
                  />
                </Box>
              }
            />
            <Divider sx={{ opacity: 0.05 }} />
            <SettingRow
              label="WASM SIMD Acceleration"
              description="Force use of vectorized math kernels where available"
              action={<Switch defaultChecked sx={{ '& .MuiSwitch-switchBase.Mui-checked': { color: qfd?.quantum }, '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': { bgcolor: qfd?.quantum } }} />}
            />
            <Divider sx={{ opacity: 0.05 }} />
            <SettingRow
              label="HFT Mode"
              description="Enable ultra-low latency routing for delta-hedging (Sub-10ms)"
              action={<Switch sx={{ '& .MuiSwitch-switchBase.Mui-checked': { color: qfd?.nebula }, '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': { bgcolor: qfd?.nebula } }} />}
            />
          </SettingsSection>

          <SettingsSection title="API Connectivity" icon={<Wifi />}>
            <Stack spacing={3}>
              <TextField
                fullWidth
                label="Polygon RPC URL"
                defaultValue="wss://polygon-mainnet.g.alchemy.com/v2/..."
                variant="outlined"
                slotProps={{ input: { sx: { fontFamily: 'JetBrains Mono', fontSize: '0.85rem' } } }}
              />
              <TextField
                fullWidth
                label="Arbitrum RPC URL"
                defaultValue="https://arb-mainnet.g.alchemy.com/v2/..."
                variant="outlined"
                slotProps={{ input: { sx: { fontFamily: 'JetBrains Mono', fontSize: '0.85rem' } } }}
              />
              <SettingRow
                label="Multicall Batch Size"
                description="Max number of contract calls per JSON-RPC request"
                action={
                  <TextField select size="small" defaultValue={50} sx={{ width: 100 }}>
                    {[20, 50, 100, 200].map(v => <MenuItem key={v} value={v}>{v}</MenuItem>)}
                  </TextField>
                }
              />
            </Stack>
          </SettingsSection>
        </Grid>

        <Grid size={{ xs: 12, lg: 4 }}>
          <SettingsSection title="Vault Security" icon={<Security />}>
            <Stack spacing={4}>
              <Box sx={{ p: 2, bgcolor: alpha('#f43f5e', 0.05), border: `1px solid ${alpha('#f43f5e', 0.2)}`, borderRadius: 3 }}>
                <Stack direction="row" spacing={1.5} sx={{ mb: 1.5 }}>
                  <ShieldMoon sx={{ color: '#f43f5e' }} />
                  <Typography variant="body2" sx={{ fontWeight: 800, color: '#f43f5e' }}>2FA Required</Typography>
                </Stack>
                <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 2 }}>
                  Your private keys are currently encrypted with AES-256 and protected by MFA.
                </Typography>
                <Button fullWidth variant="outlined" color="error" sx={{ fontWeight: 900, borderRadius: 2 }}>
                  ROTATE KEY VAULT
                </Button>
              </Box>
              
              <Box>
                <Typography variant="overline" sx={{ fontWeight: 900, color: 'text.disabled', letterSpacing: '0.1em' }}>
                  Connected Wallet
                </Typography>
                <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mt: 1, p: 2, bgcolor: alpha('#fff', 0.03), borderRadius: 3, border: `1px solid ${alpha('#fff', 0.05)}` }}>
                  <AccountBalanceWallet sx={{ color: qfd?.electrum }} />
                  <Typography variant="body2" sx={{ fontFamily: 'JetBrains Mono', fontWeight: 600 }}>0x4f2d...E92b</Typography>
                </Stack>
              </Box>

              <Button
                fullWidth
                variant="contained"
                sx={{
                  py: 1.5,
                  borderRadius: 3,
                  fontWeight: 900,
                  bgcolor: qfd?.quantum,
                  color: '#000',
                  '&:hover': { bgcolor: alpha(qfd?.quantum ?? '#00FFFF', 0.8) }
                }}
              >
                SAVE GLOBAL CONFIG
              </Button>
            </Stack>
          </SettingsSection>
        </Grid>
      </Grid>
    </Container>
  );
};

export default SettingsPage;
