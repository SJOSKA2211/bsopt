
import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

import mainTheme from './theme/mainTheme';
import App from './App'; 
import Login from './pages/Login';
import Register from './pages/Register';
import ForgotPassword from './pages/ForgotPassword';
import ResetPassword from './pages/ResetPassword';
import Analysis from './pages/Analysis';
import Dashboard from './pages/Dashboard';
import ProtectedRoute from './components/common/ProtectedRoute'; 
import { MFASetupComponent, MFADisable } from './components/auth/MFAComponents';
import SecuritySettings from './pages/SecuritySettings'; // Import SecuritySettings

// Placeholder for other pages if needed
// import NotFoundPage from './pages/NotFound'; 

export const AppRoutes = () => {
  return (
    <Router>
      <Routes>
        {/* Public Routes */}
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/forgot-password" element={<ForgotPassword />} />
        <Route path="/reset-password/:token" element={<ResetPassword />} /> 

        {/* MFA Setup Route - protected */}
        <Route path="/mfa/setup" element={
          <ProtectedRoute>
            <MFASetupComponent />
          </ProtectedRoute>
        } />

        {/* Protected Routes Group */}
        <Route
          element={
            <ProtectedRoute>
              <App /> {/* App component might contain layout, header, etc. */}
            </ProtectedRoute>
          }
        >
          <Route path="/" element={<Dashboard />} /> {/* Default to Dashboard */}
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/analysis" element={<Analysis />} />
          
          {/* Settings Section */}
          <Route path="/settings/security" element={<SecuritySettings />} /> {/* Security Settings */}
          {/* Nested route for MFA Disable, conceptually under settings */}
          <Route path="/settings/mfa/disable" element={<MFADisable />} /> 
          
          {/* Add other protected routes here */}
        </Route>

        {/* Catch-all or 404 route */}
        {/* <Route path="*" element={<NotFoundPage />} /> */}
      </Routes>
    </Router>
  );
};

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <ThemeProvider theme={mainTheme}>
      <CssBaseline />
      <AppRoutes />
    </ThemeProvider>
  </React.StrictMode>
);
