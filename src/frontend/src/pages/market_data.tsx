import React, { useState, useEffect } from 'react';
// Import Apollo Client hooks and gql tag
import { gql } from '@apollo/client'; // gql is needed for defining queries

// Import custom GraphQL hooks
import { useFetchDataGQL } from '../hooks/api'; // Assuming hooks are now GraphQL-specific

// --- GraphQL Queries ---
// Query to fetch historical market data
const GET_HISTORICAL_MARKET_DATA_QUERY = gql` # Renamed for clarity
  query GetHistoricalData($symbol: String!, $startDate: String!, $endDate: String!) {
    historicalData(symbol: $symbol, startDate: $startDate, endDate: $end) { # Assuming query structure and arguments
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
  // Default to last 30 days for start date
  const [startDate, setStartDate] = useState(new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]); 
  const [endDate, setEndDate] = useState(new Date().toISOString().split('T')[0]);

  // Fetch historical data using GraphQL hook
  // Pass query and variables. Skip query if parameters are not ready.
  const { data: historicalData, loading, error, refetch } = useFetchDataGQL<any>(GET_HISTORICAL_MARKET_DATA_QUERY, {
    variables: { symbol, startDate, endDate },
    skip: !symbol || !startDate || !endDate, // Only run query if all parameters are valid
  });

  const handleFetchData = () => {
    // Manually trigger refetch with updated variables
    refetch({ variables: { symbol, startDate, endDate } });
  };

  // Extracting data points from the fetched result
  // Assumes the query returns data under a key matching the query name (e.g., historicalData)
  const marketDataPoints = historicalData?.historicalData || []; 

  return (
    <div>
      <h1>Market Data</h1>

      {/* Controls for fetching data */}
      <div>
        <input
          type="text"
          placeholder="Symbol (e.g., AAPL)"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
        />
        <input
          type="date"
          value={startDate}
          onChange={(e) => setStartDate(e.target.value)}
        />
        <input
          type="date"
          value={endDate}
          onChange={(e) => setEndDate(e.target.value)}
        />
        <button onClick={handleFetchData} disabled={!symbol || !startDate || !endDate}>
          Fetch Data
        </button>
      </div>

      {/* Displaying Data */}
      {loading && <p>Loading market data...</p>}
      {error && <p>Error loading market data: {error}</p>}
      
      {marketDataPoints && marketDataPoints.length > 0 ? (
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th>Open</th>
              <th>High</th>
              <th>Low</th>
              <th>Close</th>
              <th>Volume</th>
            </tr>
          </thead>
          <tbody>
            {marketDataPoints.map((point) => (
              <tr key={point.date}>
                <td>{point.date}</td>
                <td>{point.open.toFixed(2)}</td>
                <td>{point.high.toFixed(2)}</td>
                <td>{point.low.toFixed(2)}</td>
                <td>{point.close.toFixed(2)}</td>
                <td>{point.volume.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (!loading && !error && <p>No market data available for the selected criteria.</p>)}
    </div>
  );
};

export default MarketDataPage;
