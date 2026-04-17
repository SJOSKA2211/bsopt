import React, { useState, useEffect } from 'react';
// Import Apollo Client hooks and gql tag
import { useQuery, useMutation, gql } from '@apollo/client';

// Import custom GraphQL hooks
import { useFetchDataGQL, useMutateDataGQL } from '../hooks/api'; 

// --- GraphQL Queries and Mutations ---
const GET_PORTFOLIOS_FOR_TRADES = gql`
  query GetPortfoliosForTrades {
    portfolios { # Assuming a query to get portfolios for the current user
      id
      name
      cash
    }
  }
`;

const CREATE_TRADE_MUTATION = gql`
  mutation CreateTrade($portfolioId: String!, $symbol: String!, $quantity: Float!, $price: Float!, $side: String!, $orderType: String!) {
    createTrade(portfolioId: $portfolioId, symbol: $symbol, quantity: $quantity, price: $price, side: $side, orderType: $orderType) { # Assuming mutation structure
      id
      portfolio_id
      symbol
      quantity
      price
      side
      order_type
      status
      timestamp
    }
  }
`;

// --- Page Component ---
const TradePage = () => {
  const [selectedPortfolioId, setSelectedPortfolioId] = useState<string>('');
  const [symbol, setSymbol] = useState('');
  const [quantity, setQuantity] = useState<number>(0);
  const [price, setPrice] = useState<number>(0);
  const [side, setSide] = useState<'buy' | 'sell'>('buy');
  const [orderType, setOrderType] = useState<string>('market');

  // Fetch portfolios using GraphQL hook
  const { data: portfolioListData, loading: portfoliosLoading, error: portfoliosError, refetch: refetchPortfolios } = useFetchDataGQL<any>(GET_PORTFOLIOS_FOR_TRADES);

  // Mutation hook for creating trades
  const { mutate: createTrade, data: createdTradeData, loading: creatingTrade, error: createTradeError } = useMutateDataGQL<any>(CREATE_TRADE_MUTATION);

  const portfolios = portfolioListData?.portfolios || [];

  useEffect(() => {
    if (portfolios && portfolios.length > 0 && !selectedPortfolioId) {
      setSelectedPortfolioId(portfolios[0].id); // Auto-select the first portfolio
    }
  }, [portfolios, selectedPortfolioId]);

  const handleTradeSubmit = async () => {
    if (!selectedPortfolioId || !symbol || quantity <= 0 || price <= 0) {
      alert('Please fill in all required trade details.');
      return;
    }

    try {
      const tradeInput = {
        portfolioId: selectedPortfolioId,
        symbol,
        quantity,
        price,
        side,
        orderType
      };
      
      const response = await createTrade({
        variables: { 
          portfolioId: selectedPortfolioId, 
          symbol, 
          quantity, 
          price, 
          side, 
          orderType 
        },
        refetchQueries: [{ query: GET_PORTFOLIOS_FOR_TRADES }], // Refetch portfolio list on trade submission (optional)
      });
      alert(`Trade submitted successfully! ID: ${response.createTrade.id}`);
      // Reset form or navigate away
      setSymbol('');
      setQuantity(0);
      setPrice(0);
    } catch (err: any) {
      console.error("Failed to submit trade:", err);
      alert(`Error submitting trade: ${err.message}`);
    }
  };

  if (portfoliosLoading) return <p>Loading portfolios...</p>;
  if (portfoliosError) return <p>Error loading portfolios: {portfoliosError}</p>;

  return (
    <div>
      <h1>Trade Execution</h1>

      {/* Portfolio Selection */}
      <div>
        <label htmlFor="portfolio-select">Select Portfolio:</label>
        <select
          id="portfolio-select"
          value={selectedPortfolioId}
          onChange={(e) => setSelectedPortfolioId(e.target.value)}
        >
          {portfolios.map((p) => (
            <option key={p.id} value={p.id}>{p.name} (${p.cash.toFixed(2)})</option>
          ))}
        </select>
      </div>

      {/* Trade Order Form */}
      <div>
        <h2>New Trade Order</h2>
        <input type="text" placeholder="Symbol (e.g., AAPL)" value={symbol} onChange={(e) => setSymbol(e.target.value)} />
        <input type="number" placeholder="Quantity" value={quantity} onChange={(e) => setQuantity(parseFloat(e.target.value) || 0)} />
        <input type="number" placeholder="Price" value={price} onChange={(e) => setPrice(parseFloat(e.target.value) || 0)} />
        
        <select value={side} onChange={(e) => setSide(e.target.value as 'buy' | 'sell')}>
          <option value="buy">Buy</option>
          <option value="sell">Sell</option>
        </select>
        
        <select value={orderType} onChange={(e) => setOrderType(e.target.value)}>
          <option value="market">Market</option>
          <option value="limit">Limit</option>
        </select>

        <button onClick={handleTradeSubmit} disabled={creatingTrade}>
          {creatingTrade ? 'Submitting...' : 'Submit Trade'}
        </button>
        {createTradeError && <p style={{ color: 'red' }}>Error: {createTradeError}</p>}
      </div>

      {/* TODO: Display recent trades for the selected portfolio */}
    </div>
  );
};

export default TradePage;
