import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/layout/Layout';
import DashboardPage from './pages/dashboard/DashboardPage';
import PortfolioPage from './pages/portfolio';

// Import Apollo Client hooks and gql tag
import { useQuery, useMutation, gql } from '@apollo/client';

// --- GraphQL Queries and Mutations ---
const GET_PORTFOLIOS = gql`
  query GetPortfolios {
    portfolios { # Assuming a 'portfolios' query exists at the backend
      id
      name
      cash
      user_id
      created_at
    }
  }
`;

const CREATE_PORTFOLIO = gql`
  mutation CreatePortfolio($name: String!, $cash: Float!) {
    createPortfolio(name: $name, cash: $cash) { # Assuming mutation arguments match backend schema
      id
      name
      cash
      user_id
      created_at
    }
  }
`;

const UPDATE_PORTFOLIO = gql`
  mutation UpdatePortfolio($id: String!, $data: PortfolioUpdateInput!) { # Assuming input types
    updatePortfolio(id: $id, data: $data) {
      id
      name
      cash
      updated_at
    }
  }
`;

// --- Page Component ---
const PortfolioPage = () => {
  const [newPortfolioName, setNewPortfolioName] = useState('');
  const [newPortfolioCash, setNewPortfolioCash] = useState<number>(0);
  
  // Fetch portfolios using Apollo useQuery hook
  const { data: portfolioData, loading, error, refetch } = useQuery(GET_PORTFOLIOS);

  // Mutation hook for creating portfolios
  const [createPortfolioMutation, { loading: creatingPortfolio, error: createError }] = useMutation(CREATE_PORTFOLIO, {
      refetchQueries: [{ query: GET_PORTFOLIOS }], // Automatically refetch list after mutation
  });

  // Mutation hook for updating portfolios
  const [updatePortfolioMutation] = useMutation(UPDATE_PORTFOLIO);

  const handleCreatePortfolio = async () => {
    if (!newPortfolioName || newPortfolioCash <= 0) {
      alert('Please enter a valid portfolio name and cash amount.');
      return;
    }
    
    try {
      await createPortfolioMutation({
        variables: { name: newPortfolioName, cash: newPortfolioCash },
      });
      alert(`Portfolio "${newPortfolioName}" created successfully!`);
      setNewPortfolioName('');
      setNewPortfolioCash(0);
      // refetch is handled by refetchQueries option in useMutation
    } catch (err: any) {
      console.error("Failed to create portfolio:", err);
      alert(`Error creating portfolio: ${err.message}`);
    }
  };

  // TODO: Implement update logic using updatePortfolioMutation hook

  if (loading) return <p>Loading portfolios...</p>;
  if (error) return <p>Error loading portfolios: {error.message}</p>;

  const portfolios = portfolioData?.portfolios || []; 

  return (
    <div>
      <h1>Portfolios</h1>

      {/* Create New Portfolio Form */}
      <div>
        <h2>Create New Portfolio</h2>
        <input
          type="text"
          placeholder="Portfolio Name"
          value={newPortfolioName}
          onChange={(e) => setNewPortfolioName(e.target.value)}
        />
        <input
          type="number"
          placeholder="Initial Cash"
          value={newPortfolioCash}
          onChange={(e) => setNewPortfolioCash(parseFloat(e.target.value) || 0)}
        />
        <button onClick={handleCreatePortfolio} disabled={!newPortfolioName || newPortfolioCash <= 0 || creatingPortfolio}>
          {creatingPortfolio ? 'Creating...' : 'Create Portfolio'}
        </button>
        {createError && <p style={{ color: 'red' }}>Error: {createError.message}</p>}
      </div>

      {/* Portfolio List */}
      <div>
        <h2>Your Portfolios</h2>
        {portfolios && portfolios.length > 0 ? (
          <ul>
            {portfolios.map((p) => (
              <li key={p.id}>
                {p.name} - Cash: ${p.cash} (ID: {p.id})
                {/* TODO: Add links to view/edit portfolio details */}
              </li>
            ))}
          </ul>
        ) : (
          <p>No portfolios found.</p>
        )}
      </div>
    </div>
  );
};

export default PortfolioPage;
