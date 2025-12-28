import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Container, Typography, TextField, Button, Box, Paper, Alert, CircularProgress, Stepper, Step, StepLabel, StepContent, ButtonGroup } from '@mui/material';
import apiClient from '@/utils/apiClient';
import { QRCodeSVG } from 'qrcode.react'; // For displaying QR code
import { AxiosError } from 'axios';

interface ErrorResponse {
  message?: string;
  detail?: string | unknown;
}

// Assume these are defined in your backend schemas or types
interface MFASetupResponse {
  secret: string;
  qr_code_uri: string;
  backup_codes: string[];
}

// --- MFA Setup Component (Already created in previous step) ---
// Keeping it here for completeness, though it would typically be in its own file.
const MFASetupComponent: React.FC = () => {
  const [step, setStep] = useState(0);
  const [mfaSecret, setMfaSecret] = useState<string | null>(null);
  const [qrCodeUri, setQrCodeUri] = useState<string | null>(null);
  const [backupCodes, setBackupCodes] = useState<string[]>([]);
  const [mfaCode, setMfaCode] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  
  const navigate = useNavigate();

  const handleSetupMFA = async () => {
    setLoading(true); setError(null); setSuccessMessage(null);
    try {
      const response = await apiClient.post<MFASetupResponse>('/auth/mfa/setup');
      setMfaSecret(response.data.secret);
      setQrCodeUri(response.data.qr_code_uri);
      setBackupCodes(response.data.backup_codes);
      setStep(1); 
    } catch (err: unknown) { // Changed from AxiosError to unknown
      console.error('MFA setup failed:', err);
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
    } finally {
      setLoading(false);
    }
  };

  const handleVerifyMFA = async () => {
    if (!mfaSecret) { setError('MFA setup incomplete.'); return; }
    setLoading(true); setError(null); setSuccessMessage(null);
    try {
      await apiClient.post('/auth/mfa/verify', { code: mfaCode });
      setSuccessMessage('MFA enabled successfully! Please save your backup codes.');
      setStep(2); 
    } catch (err: unknown) { // Changed from AxiosError to unknown
      console.error('MFA verification failed:', err);
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
    } finally {
      setLoading(false);
    }
  };

  const handleGoBack = () => {
    setStep(0); setMfaSecret(null); setQrCodeUri(null); setBackupCodes([]); setMfaCode(''); setError(null); setSuccessMessage(null);
  };

  const handleFinish = () => {
    navigate('/dashboard'); // Redirect to dashboard or settings
  };

  return (
    <Container maxWidth="sm" sx={{ mt: 8 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Multi-Factor Authentication (MFA) Setup
        </Typography>
        <Stepper activeStep={step} orientation="vertical" sx={{ mb: 3 }}>
          <Step key="initiate">
            <StepLabel>Initiate MFA Setup</StepLabel>
            <StepContent>
              <Typography variant="body2" sx={{ mb: 2 }}>Enable an extra layer of security for your account.</Typography>
              <Button variant="contained" onClick={handleSetupMFA} disabled={loading}>
                {loading ? <CircularProgress size={24} /> : 'Start Setup'}
              </Button>
            </StepContent>
          </Step>
          <Step key="verify" completed={step > 1}>
            <StepLabel>Verify MFA Code</StepLabel>
            <StepContent>
              {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
              {qrCodeUri && (
                <Box sx={{ mb: 2, textAlign: 'center' }}>
                  <Typography variant="subtitle1" gutterBottom>Scan this QR Code with your authenticator app:</Typography>
                  <QRCodeSVG value={qrCodeUri} size={128} level="H" />
                  <Typography variant="caption" display="block" sx={{ mt: 1 }}>Secret: {mfaSecret}</Typography>
                </Box>
              )}
              <TextField
                margin="normal" required fullWidth id="mfaCode" label="Enter 6-Digit MFA Code" name="mfaCode"
                autoComplete="one-time-code" value={mfaCode} onChange={(e) => setMfaCode(e.target.value)}
                sx={{ mb: 2 }} disabled={loading || !!successMessage}
              />
              <ButtonGroup variant="contained" fullWidth>
                <Button onClick={handleVerifyMFA} disabled={loading || !!successMessage}>
                  {loading ? <CircularProgress size={24} /> : 'Verify Code & Enable MFA'}
                </Button>
                <Button onClick={handleGoBack} disabled={loading || !!successMessage}>Back</Button>
              </ButtonGroup>
            </StepContent>
          </Step>
          <Step key="complete" completed={step > 1}>
            <StepLabel>Setup Complete</StepLabel>
            <StepContent>
              {successMessage && (
                <Alert severity="success" sx={{ mb: 2 }}>
                  {successMessage}
                </Alert>
              )}
              <Typography variant="body2" sx={{ mb: 2 }}>
                Your backup codes are listed below. Store them securely, as they can be used to log in if you lose access to your authenticator app.
              </Typography>
              {backupCodes.length > 0 && (
                <Paper variant="outlined" sx={{ p: 2, mb: 2, bgcolor: '#f9f9f9' }}>
                  <Typography variant="subtitle1" gutterBottom>Backup Codes:</Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {backupCodes.map((code, index) => (
                      <Typography key={index} variant="body2" sx={{ fontFamily: 'monospace', p: 0.5, border: '1px solid #ccc', borderRadius: '4px' }}>
                        {code}
                      </Typography>
                    ))}
                  </Box>
                </Paper>
              )}
              <Button variant="contained" onClick={handleFinish}>Done</Button>
            </StepContent>
          </Step>
        </Stepper>
      </Paper>
    </Container>
  );
};

// --- MFA Disable Component ---
const MFADisable: React.FC = () => {
    const [mfaCode, setMfaCode] = useState('');
    const [error, setError] = useState<string | null>(null);
    const [successMessage, setSuccessMessage] = useState<string | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const navigate = useNavigate();

    const handleDisableMFA = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true); setError(null); setSuccessMessage(null);
        try {
            await apiClient.post('/auth/mfa/disable', { code: mfaCode });
            setSuccessMessage('MFA has been disabled successfully.');
            // Potentially redirect or refresh user state
            setTimeout(() => navigate('/dashboard'), 3000);
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
                {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
                {successMessage && <Alert severity="success" sx={{ mb: 2 }}>{successMessage}</Alert>}
                <Box component="form" onSubmit={handleDisableMFA} noValidate sx={{ mt: 1 }}>
                    <TextField
                        margin="normal" required fullWidth id="mfaCode" label="Enter 6-Digit MFA Code" name="mfaCode"
                        autoComplete="one-time-code" value={mfaCode} onChange={(e) => setMfaCode(e.target.value)}
                        sx={{ mb: 2 }} disabled={loading || !!successMessage}
                    />
                    <Button
                        type="submit" fullWidth variant="contained" color="primary"
                        sx={{ mt: 3, mb: 2 }} disabled={loading || !!successMessage}
                    >
                        {loading ? <CircularProgress size={24} /> : 'Disable MFA'}
                    </Button>
                </Box>
            </Paper>
        </Container>
    );
};

// --- MFA Login Step Component (Conditional) ---
// This would be integrated into the Login flow, prompted when login response indicates requires_mfa=true
const MFALoginStep: React.FC<{ onVerify: (code: string) => void; error: string | null; loading: boolean }> = 
    ({ onVerify, error, loading }) => {
    const [mfaCode, setMfaCode] = useState('');
    return (
        <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle1" gutterBottom>Enter your MFA code:</Typography>
            {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
            <TextField
                margin="normal" required fullWidth id="mfaCode" label="MFA Code" name="mfaCode"
                autoComplete="one-time-code" value={mfaCode} onChange={(e) => setMfaCode(e.target.value)}
                sx={{ mb: 2 }}
            />
            <Button
                variant="contained" fullWidth onClick={() => onVerify(mfaCode)} disabled={loading}
            >
                {loading ? <CircularProgress size={24} /> : 'Verify MFA'}
            </Button>
        </Box>
    );
};

export { MFASetupComponent, MFADisable, MFALoginStep };
