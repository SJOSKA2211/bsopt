import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, gql } from '@apollo/client'; // Import GraphQL hooks
import { ApolloError } from '@apollo/client'; // Import ApolloError for specific error handling

// Import custom GraphQL hooks
import { useFetchDataGQL, useMutateDataGQL } from '../hooks/api'; 

// --- GraphQL Queries and Mutations ---
const GET_HISTORICAL_MARKET_DATA_QUERY = gql`
  query GetHistoricalData($symbol: String!, $startDate: String!, $endDate: String!) {
    historicalData(symbol: $symbol, startDate: $startDate, endDate: $end) { 
      date
      open
      high
      low
      close
      volume
    }
  }
`;

// --- Page Component ---
const MarketDataPage = () => {
  const [symbol, setSymbol] = useState('AAPL');
  const [startDate, setStartDate] = useState(new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]); 
  const [endDate, setEndDate] = useState(new Date().toISOString().split('T')[0]);

  // Fetch historical data using GraphQL hook
  const { data: historicalData, loading, error, refetch } = useFetchDataGQL<any>(GET_HISTORICAL_MARKET_DATA_QUERY, {
    variables: { symbol, startDate, endDate },
    skip: !symbol || !startDate || !endDate, // Only run query if all parameters are ready
  });

  const handleFetchData = () => {
    refetch({ variables: { symbol, startDate, endDate } }); // Refetch with updated variables
  };

  const marketDataPoints = historicalData?.historicalData || []; 

  return (
    <div className="min-h-screen p-6 bg-bento-bg text-white"> {/* Applying global dark background */}
      <h1 className="text-3xl font-bold mb-6">Market Data</h1>

      {/* Controls for fetching data */}
      <div className="mb-8 p-6 bg-gray-800 bg-opacity-75 backdrop-blur-md border border-gray-700 rounded-xl shadow-lg">
        <h2 className="text-2xl font-semibold mb-4">Historical Data Query</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4 items-end">
          <div className="flex flex-col gap-1">
            <label htmlFor="symbol" className="text-sm font-medium text-gray-400">Symbol</label>
            <input
              id="symbol"
              type="text"
              placeholder="Symbol (e.g., AAPL)"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className="p-3 rounded-lg bg-gray-700 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-mint"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label htmlFor="start-date" className="text-sm font-medium text-gray-400">Start Date</label>
            <input
              id="start-date"
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="p-3 rounded-lg bg-gray-700 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-mint"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label htmlFor="end-date" className="text-sm font-medium text-gray-400">End Date</label>
            <input
              id="end-date"
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="p-3 rounded-lg bg-gray-700 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-mint"
            />
          </div>
          <button 
            onClick={handleFetchData} 
            disabled={!symbol || !startDate || !endDate}
            className="px-6 py-3 bg-mint text-bento-bg font-semibold rounded-lg shadow-md hover:bg-opacity-90 disabled:opacity-50 transition-colors duration-200"
          >
            Fetch Data
          </button>
        </div>
      </div>

      {/* Displaying Data */}
      <div className="p-6 bg-gray-800 bg-opacity-75 backdrop-blur-md border border-gray-700 rounded-xl shadow-lg">
        <h2 className="text-2xl font-semibold mb-4">Historical Data</h2>
        {loading && <p className="text-lg animate-pulse">Loading market data...</p>}
        {error && <p className="text-red-500 text-lg">Error loading market data: {error}</p>}
        
        {marketDataPoints && marketDataPoints.length > 0 ? (
          <div className="overflow-x-auto"> 
            <table className="min-w-full table-auto">
              <thead className="bg-gray-700 bg-opacity-75">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Date</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Open</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">High</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Low</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Close</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Volume</th>
                </tr>
              </thead>
              <tbody className="bg-bento-bg divide-y divide-gray-700">
                {marketDataPoints.map((point) => (
                  <tr key={point.date} className="hover:bg-gray-700 hover:bg-opacity-50 transition-colors duration-150">
                    <td className="px-4 py-3 whitespace-nowrap">{point.date}</td>
                    <td className="px-4 py-3 whitespace-nowrap">{point.open.toFixed(2)}</td>
                    <td className="px-4 py-3 whitespace-nowrap">{point.high.toFixed(2)}</td>
                    <td className="px-4 py-3 whitespace-nowrap">{point.low.toFixed(2)}</td>
                    <td className="px-4 py-3 whitespace-nowrap">{point.close.toFixed(2)}</td>
                    <td className="px-4 py-3 whitespace-nowrap">{point.volume.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (!loading && !error && <p>No market data available for the selected criteria.</p>)}
      </div>
    </div>
  );
};

export default MarketDataPage;
