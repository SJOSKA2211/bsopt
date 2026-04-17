import React, { useState, useEffect } from 'react';
// Note: Removed react-router-dom and Layout imports as they are not directly used in this file's context,
// but should be present in the main App.tsx.

// Import Apollo Client hooks and gql tag
import { useQuery, useMutation, gql } from '@apollo/client';

// Import custom hooks for API interaction (now GraphQL based)
import { useFetchDataGQL, useMutateDataGQL } from '../hooks/api'; // Assuming hooks are now GraphQL-specific

// --- GraphQL Queries and Mutations ---
const GET_PORTFOLIOS = gql`
  query GetPortfolios {
    portfolios { 
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
    createPortfolio(name: $name, cash: $cash) { 
      id
      name
      cash
      user_id
      created_at
    }
  }
`;

const UPDATE_PORTFOLIO = gql`
  mutation UpdatePortfolio($id: String!, $data: PortfolioUpdateInput!) { 
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
  
  // Fetch portfolios using the GraphQL hook
  const { data: portfolioData, loading, error, refetch } = useFetchDataGQL<any>(GET_PORTFOLIOS);

  // Mutation hook for creating portfolios
  const [createPortfolioMutation, { loading: creatingPortfolio, error: createError }] = useMutateDataGQL(CREATE_PORTFOLIO);

  // Mutation hook for updating portfolios
  const [updatePortfolioMutation] = useMutateDataGQL(UPDATE_PORTFOLIO);

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
      // refetch is handled by refetchQueries option in useMutation or automatically by useQuery setup
      // If not automatic, uncomment: await refetch();
    } catch (err: any) {
      console.error("Failed to create portfolio:", err);
      alert(`Error creating portfolio: ${err.message}`);
    }
  };

  const handleUpdatePortfolio = async (portfolioId: string, updatedCash: number) => {
      try {
          await updatePortfolioMutation({
              variables: { id: portfolio_id, data: { cash: updatedPortfolioCash } }, // Assuming update mutation variables structure
          });
          alert(`Portfolio ${portfolioId} updated successfully!`);
          refetch(); // Refetch the list to show changes
      } catch (err: any) {
          console.error(`Failed to update portfolio ${portfolio_id}:`, err);
          alert(`Error updating portfolio: ${err.message}`);
      }
  };

  if (loading) return <p>Loading portfolios...</p>;
  if (error) return <p>Error loading portfolios: {error}</p>;

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
        {createError && <p style={{ color: 'red' }}>Error: {createError}</p>}
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
                <button onClick={() => handleUpdatePortfolio(p.id, p.cash + 100)}>Update Cash (Simulated)</button> 
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
