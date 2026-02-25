import { useState } from "react";
import { Box, Button, TextField, Typography, Container, Paper, Alert } from "@mui/material";
import { authClient } from "../../lib/auth-client";

export default function SignIn() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);

  const signIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setSuccess(false);

    await authClient.signIn.email({ email, password }, {
      onRequest: () => setLoading(true),
      onSuccess: () => {
        setLoading(false);
        setSuccess(true);
      },
      onError: (ctx) => {
        setLoading(false);
        setError(ctx.error.message);
      },
    });
  };

  return (
    <Container maxWidth="xs">
      <Paper elevation={3} sx={{ p: 4, mt: 8, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <Typography component="h1" variant="h5" gutterBottom>Sign In</Typography>
        <Box component="form" onSubmit={signIn} sx={{ mt: 1, width: '100%' }}>
          {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
          {success && <Alert severity="success" sx={{ mb: 2 }}>Signed in successfully!</Alert>}
          <TextField margin="normal" required fullWidth label="Email Address" autoFocus value={email} onChange={(e) => setEmail(e.target.value)} disabled={loading} />
          <TextField margin="normal" required fullWidth label="Password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} disabled={loading} />
          <Button type="submit" fullWidth variant="contained" sx={{ mt: 3, mb: 2 }} disabled={loading}>
            {loading ? "Signing in..." : "Sign In"}
          </Button>
        </Box>
      </Paper>
    </Container>
  );
}
