import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Container, Typography, TextField, Button, Box, Paper, Alert, CircularProgress, Stepper, Step, StepLabel, StepContent, ButtonGroup } from '@mui/material';
import apiClient from '@/utils/apiClient'; // Import the API client
import { AxiosError } from 'axios'; 
import { QRCodeSVG } from 'qrcode.react'; // For displaying QR code

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

const MFASetup: React.FC = () => {
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
    setLoading(true);
    setError(null);
    setSuccessMessage(null);
    try {
      const response = await apiClient.post<MFASetupResponse>('/auth/mfa/setup');
      setMfaSecret(response.data.secret);
      setQrCodeUri(response.data.qr_code_uri);
      setBackupCodes(response.data.backup_codes);
      setStep(1); // Move to verification step
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
    if (!mfaSecret) { // Should not happen if step is 1, but for safety
      setError('MFA setup incomplete.');
      return;
    }
    setLoading(true);
    setError(null);
    setSuccessMessage(null);
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
    setStep(0); // Go back to setup initiation
    setMfaSecret(null);
    setQrCodeUri(null);
    setBackupCodes([]);
    setMfaCode('');
    setError(null);
    setSuccessMessage(null);
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
          {/* Step 0: Initiate Setup */}
          <Step key="initiate">
            <StepLabel>Initiate MFA Setup</StepLabel>
            <StepContent>
              <Typography variant="body2" sx={{ mb: 2 }}>
                Enable an extra layer of security for your account.
              </Typography>
              <Button variant="contained" onClick={handleSetupMFA} disabled={loading}>
                {loading ? <CircularProgress size={24} /> : 'Start Setup'}
              </Button>
            </StepContent>
          </Step>

          {/* Step 1: Verify MFA Code */}
          <Step key="verify" completed={step > 1}>
            <StepLabel>Verify MFA Code</StepLabel>
            <StepContent>
              {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
              {qrCodeUri && (
                <Box sx={{ mb: 2, textAlign: 'center' }}>
                  <Typography variant="subtitle1" gutterBottom>Scan this QR Code with your authenticator app:</Typography>
                  <QRCodeSVG value={qrCodeUri} size={128} level="H" />
                  <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                    Secret: {mfaSecret} (Copy if needed, but QR is preferred)
                  </Typography>
                </Box>
              )}
              <TextField
                margin="normal"
                required
                fullWidth
                id="mfaCode"
                label="Enter 6-Digit MFA Code"
                name="mfaCode"
                autoComplete="one-time-code"
                value={mfaCode}
                onChange={(e) => setMfaCode(e.target.value)}
                sx={{ mb: 2 }}
                disabled={loading || !!successMessage}
              />
              <ButtonGroup variant="contained" fullWidth>
                <Button onClick={handleVerifyMFA} disabled={loading || !!successMessage}>
                  {loading ? <CircularProgress size={24} /> : 'Verify Code & Enable MFA'}
                </Button>
                <Button onClick={handleGoBack} disabled={loading || !!successMessage}>
                  Back
                </Button>
              </ButtonGroup>
            </StepContent>
          </Step>

          {/* Step 2: Completion */}
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
              <Button variant="contained" onClick={handleFinish}>
                Done
              </Button>
            </StepContent>
          </Step>
        </Stepper>
      </Paper>
    </Container>
  );
};

export default MFASetup;
