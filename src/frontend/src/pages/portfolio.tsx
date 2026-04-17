import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, gql } from '@apollo/client'; 
import { ApolloError } from '@apollo/client'; // Import ApolloError for type checking

// --- GraphQL Queries and Mutations ---
const GET_PORTFOLIOS_QUERY = gql`
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

const CREATE_PORTFOLIO_MUTATION = gql`
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

const UPDATE_PORTFOLIO_MUTATION = gql`
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
  
  // Fetch portfolios using Apollo's useQuery hook
  const { data: portfolioData, loading, error, refetch } = useQuery(GET_PORTFOLIOS_QUERY);

  // Mutation hook for creating portfolios
  const [createPortfolio, { loading: creatingPortfolio, error: createError }] = useMutation(CREATE_PORTFOLIO_MUTATION, {
      refetchQueries: [{ query: GET_PORTFOLIOS_QUERY }], // Automatically refetch list after mutation
  });

  // Mutation hook for updating portfolios
  const [updatePortfolioMutation] = useMutation(UPDATE_PORTFOLIO_MUTATION);

  const handleCreatePortfolio = async () => {
    if (!newPortfolioName || newPortfolioCash <= 0) {
      alert('Please enter a valid portfolio name and cash amount.');
      return;
    }
    
    try {
      await createPortfolio({
        variables: { name: newPortfolioName, cash: newPortfolioCash },
      });
      alert(`Portfolio "${newPortfolioName}" created successfully!`);
      setNewPortfolioName('');
      setNewPortfolioCash(0);
    } catch (err: any) {
      console.error("Failed to create portfolio:", err);
      alert(`Error creating portfolio: ${err.message}`);
    }
  };

  const handleUpdatePortfolio = async (portfolioId: string, updatedCash: number) => {
      try {
          await updatePortfolioMutation({
              variables: { id: portfolioId, data: { cash: updatedCash } },
          });
          alert(`Portfolio ${portfolioId} updated successfully!`);
          refetch(); // Refetch the list to show changes
      } catch (err: any) {
          console.error(`Failed to update portfolio ${portfolioId}:`, err);
          alert(`Error updating portfolio: ${err.message}`);
      }
  };

  if (loading) return (
    <div className="min-h-screen flex items-center justify-center bg-bento-bg text-white">
      <p className="text-lg animate-pulse">Loading portfolios...</p> {/* Added loading animation */}
    </div>
  );
  // Check for ApolloError specifically if needed, or general error message
  if (error) return (
    <div className="min-h-screen flex items-center justify-center bg-bento-bg text-white">
      <p className="text-lg text-red-500">Error loading portfolios: {error.message}</p>
    </div>
  );

  const portfolios = portfolioData?.portfolios || []; 

  return (
    <div className="container mx-auto p-6 bg-bento-bg text-white min-h-screen">
      <h1 className="text-3xl font-bold mb-6">Portfolios</h1>

      {/* Create New Portfolio Form */}
      <div className="mb-8 p-6 bg-gray-800 bg-opacity-75 backdrop-blur-md border border-gray-700 rounded-xl shadow-lg">
        <h2 className="text-2xl font-semibold mb-4">Create New Portfolio</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
          <input
            type="text"
            placeholder="Portfolio Name"
            value={newPortfolioName}
            onChange={(e) => setNewPortfolioName(e.target.value)}
            className="p-3 rounded-lg bg-gray-700 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-mint"
          />
          <input
            type="number"
            placeholder="Initial Cash"
            value={newPortfolioCash}
            onChange={(e) => setNewPortfolioCash(parseFloat(e.target.value) || 0)}
            className="p-3 rounded-lg bg-gray-700 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-mint"
          />
        </div>
        <button 
          onClick={handleCreatePortfolio} 
          disabled={!newPortfolioName || newPortfolioCash <= 0 || creatingPortfolio}
          className="px-6 py-3 bg-mint text-bento-bg font-semibold rounded-lg shadow-md hover:bg-opacity-90 disabled:opacity-50 transition-colors duration-200"
        >
          {creatingPortfolio ? 'Creating...' : 'Create Portfolio'}
        </button>
        {createError && <p className="text-red-500 mt-4">Error: {createError.message}</p>}
      </div>

      {/* Portfolio List */}
      <div>
        <h2 className="text-2xl font-semibold mb-4">Your Portfolios</h2>
        {portfolios && portfolios.length > 0 ? (
          <ul className="space-y-4">
            {portfolios.map((p) => (
              <li key={p.id} className="p-4 bg-gray-800 bg-opacity-75 backdrop-blur-md border border-gray-700 rounded-lg shadow-lg flex justify-between items-center">
                <div>
                  <p className="text-lg font-semibold">{p.name}</p>
                  <p className="text-sm text-gray-400">Cash: ${p.cash.toFixed(2)}</p>
                  <p className="text-xs text-gray-500">ID: {p.id}</p>
                </div>
                {/* TODO: Add links to view/edit portfolio details */}
                <button 
                  onClick={() => handleUpdatePortfolio(p.id, p.cash + 100)} 
                  className="px-4 py-2 bg-mint text-bento-bg font-semibold rounded-lg shadow-md hover:bg-opacity-90 transition-colors duration-200"
                >
                  Update Cash (Simulated)
                </button>
              </li>
            ))}
          </ul>
        ) : (!loading && !error && <p>No portfolios found.</p>)}
      </div>
    </div>
  );
};

export default PortfolioPage;
