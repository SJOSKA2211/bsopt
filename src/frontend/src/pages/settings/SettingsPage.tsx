import React from 'react';
import { Container, Typography, Paper, Box } from '@mui/material';

export const SettingsPage: React.FC = () => {
  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom fontWeight="bold">Settings</Typography>
      <Paper sx={{ p: 4 }}>
        <Typography variant="h6">User Preferences</Typography>
        <Box sx={{ mt: 2 }}>
            <Typography color="text.secondary">Settings implementation coming soon...</Typography>
        </Box>
      </Paper>
    </Container>
  );
};

export default SettingsPage;
