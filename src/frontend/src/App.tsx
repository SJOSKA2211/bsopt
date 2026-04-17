import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/layout/Layout';
import DashboardPage from './pages/dashboard/DashboardPage';
import PortfolioPage from './pages/portfolio'; // Import the new PortfolioPage

const App = () => (
  <BrowserRouter>
    <Layout>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/portfolios" element={<PortfolioPage />} /> {/* Add route for PortfolioPage */}
        {/* Add other routes here: /trade, /ml, /settings, etc. */}
      </Routes>
    </Layout>
  </BrowserRouter>
);

export default App;
