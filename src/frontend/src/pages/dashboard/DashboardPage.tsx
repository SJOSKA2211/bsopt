import React, { useState, useEffect } from 'react';
import { gql, useQuery } from '@apollo/client'; // Assuming Apollo Client is set up for frontend queries
import { useFetchDataGQL } from '../../hooks/api'; // Import the GraphQL fetch hook

// --- GraphQL Query ---
// Query to fetch a summary of the user's primary portfolio value (simulated net liquidation)
const GET_PORTFOLIO_SUMMARY = gql`
  query GetPortfolioSummary {
    portfolios(limit: 1) { # Fetching the first portfolio as a summary
      id
      name
      cash
      # In a real app, this might be a dedicated summary query or aggregated value
    }
  }
`;

const DashboardPage = () => {
  const [signalStatus, setSignalStatus] = useState<string>('Idle');

  // Fetch portfolio summary using GraphQL hook
  const { data: portfolioSummaryData, loading: portfolioLoading, error: portfolioError, refetch: refetchPortfolio } = useFetchDataGQL<any>(GET_PORTFOLIOS_QUERY); // Should use GET_PORTFOLIOS_QUERY or a dedicated summary query

  // Simulate real-time updates for SIGNAL_ENGINE status
  useEffect(() => {
    if (!portfolioSummaryData) return; // Don't start if no data yet

    const interval = setInterval(() => {
      // Simulate status change based on some logic or mock data
      const statuses = ['Idle', 'Running', 'Analyzing', 'Alert'];
      setSignalStatus(statuses[Math.floor(Math.random() * statuses.length)]);
    }, 5000); // Update status every 5 seconds

    return () => clearInterval(interval); // Cleanup interval on component unmount
  }, [portfolioSummaryData]); // Restart effect if portfolio data changes

  const netLiquidation = portfolioSummaryData?.portfolios?.[0]?.cash || 0; // Extract cash from the first portfolio as net liquidation proxy

  return (
    <div className="min-h-screen p-6 bg-bento-bg text-white">
      <h1 className="text-4xl font-bold mb-8">Dashboard</h1>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Net Liquidation Card */}
        <div className="col-span-1 lg:col-span-1 bento-card p-6">
          <span className="label-secondary">NET LIQUIDATION</span>
          <div className="text-4xl font-black mt-2 font-mono">
            {portfolioLoading ? '...' : `$${parseFloat(netLiquidation.toString()).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          </div>
          {portfolioError && <p className="text-red-500 text-sm mt-2">Error: {portfolioError}</p>}
        </div>

        {/* Signal Engine Card */}
        <div className="col-span-1 lg:col-span-2 bento-card h-[400px] flex flex-col justify-center items-center p-6">
          <span className="label-secondary">SIGNAL ENGINE</span>
          <p className="text-3xl font-bold my-4">Status: <span className={`font-mono ${signalStatus === 'Alert' ? 'text-red-500' : 'text-mint'}`}>{signalStatus}</span></p>
          <button 
            onClick={() => alert('Action triggered! (Simulated)')}
            className="px-6 py-3 bg-mint text-bento-bg font-semibold rounded-lg shadow-md hover:bg-opacity-90"
          >
            Trigger Signal Analysis
          </button>
        </div>
        
        {/* Add more cards/widgets here for other dashboard elements */}
        {/* Example: Recent Trades Widget */}
        <div className="col-span-1 lg:col-span-1 bento-card p-6">
            <span className="label-secondary">RECENT ACTIVITY</span>
            <p className="text-lg mt-4">Placeholder for recent trades or actions.</p>
            {/* TODO: Fetch and display recent trades */}
        </div>

      </div>
    </div>
  );
};

export default DashboardPage;
