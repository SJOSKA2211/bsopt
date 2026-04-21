import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, gql } from '@apollo/client'; // Import GraphQL hooks
import { ApolloError } from '@apollo/client'; // Import ApolloError for specific error handling

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
  const { data: portfolioListData, loading: portfoliosLoading, error: portfoliosError, refetch: refetchPortfolios } = useQuery(GET_PORTFOLIOS_FOR_TRADES);

  // Mutation hook for creating trades
  const [createTradeMutation, { loading: creatingTrade, error: createTradeError }] = useMutation(CREATE_TRADE_MUTATION, {
      refetchQueries: [{ query: GET_PORTFOLIOS_FOR_TRADES }], // Refetch portfolio list after trade, or use a specific trade list query
  });

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
      
      const response = await createTradeMutation({
        variables: { 
          portfolioId: selectedPortfolioId, 
          symbol, 
          quantity, 
          price, 
          side, 
          orderType 
        },
      });
      alert(`Trade submitted successfully! ID: ${response.data.createTrade.id}`);
      // Reset form or navigate away
      setSymbol('');
      setQuantity(0);
      setPrice(0);
    } catch (err: any) {
      console.error("Failed to submit trade:", err);
      alert(`Error submitting trade: ${err.message}`);
    }
  };

  if (portfoliosLoading) return (
    <div className="min-h-screen flex items-center justify-center bg-bento-bg text-white">
      <p className="text-lg animate-pulse">Loading portfolios...</p>
    </div>
  );
  if (portfoliosError) return (
    <div className="min-h-screen flex items-center justify-center bg-bento-bg text-white">
      <p className="text-lg text-red-500">Error loading portfolios: {portfoliosError.message}</p>
    </div>
  );

  return (
    <div className="container mx-auto p-6 bg-bento-bg text-white min-h-screen">
      <h1 className="text-3xl font-bold mb-6">Trade Execution</h1>

      {/* Portfolio Selection */}
      <div className="mb-8 p-6 bg-gray-800 bg-opacity-75 backdrop-blur-md border border-gray-700 rounded-xl shadow-lg">
        <label htmlFor="portfolio-select" className="block text-lg font-semibold mb-3">Select Portfolio:</label>
        <select
          id="portfolio-select"
          value={selectedPortfolioId}
          onChange={(e) => setSelectedPortfolioId(e.target.value)}
          className="p-3 rounded-lg bg-gray-700 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-mint w-full"
        >
          {portfolios.map((p) => (
            <option key={p.id} value={p.id}>{p.name} (${p.cash.toFixed(2)})</option>
          ))}
        </select>
      </div>

      {/* Trade Order Form */}
      <div className="p-6 bg-gray-800 bg-opacity-75 backdrop-blur-md border border-gray-700 rounded-xl shadow-lg">
        <h2 className="text-2xl font-semibold mb-4">New Trade Order</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
          <div className="flex flex-col gap-2">
            <label htmlFor="symbol" className="text-sm font-medium">Symbol</label>
            <input
              id="symbol"
              type="text"
              placeholder="Symbol (e.g., AAPL)"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className="p-3 rounded-lg bg-gray-700 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-mint"
            />
          </div>
          <div className="flex flex-col gap-2">
            <label htmlFor="quantity" className="text-sm font-medium">Quantity</label>
            <input
              id="quantity"
              type="number"
              placeholder="Quantity"
              value={quantity}
              onChange={(e) => setQuantity(parseFloat(e.target.value) || 0)}
              className="p-3 rounded-lg bg-gray-700 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-mint"
            />
          </div>
          <div className="flex flex-col gap-2">
            <label htmlFor="price" className="text-sm font-medium">Price</label>
            <input
              id="price"
              type="number"
              placeholder="Price"
              value={price}
              onChange={(e) => setPrice(parseFloat(e.target.value) || 0)}
              className="p-3 rounded-lg bg-gray-700 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-mint"
            />
          </div>
          
          <div className="flex flex-col gap-2">
            <label htmlFor="side" className="text-sm font-medium">Side</label>
            <select id="side" value={side} onChange={(e) => setSide(e.target.value as 'buy' | 'sell')} className="p-3 rounded-lg bg-gray-700 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-mint">
              <option value="buy">Buy</option>
              <option value="sell">Sell</option>
            </select>
          </div>
          
          <div className="flex flex-col gap-2">
            <label htmlFor="orderType" className="text-sm font-medium">Order Type</label>
            <select id="orderType" value={orderType} onChange={(e) => setOrderType(e.target.value)} className="p-3 rounded-lg bg-gray-700 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-mint">
              <option value="market">Market</option>
              <option value="limit">Limit</option>
            </select>
          </div>
        </div>

        <button 
          onClick={handleTradeSubmit} 
          disabled={creatingTrade}
          className="px-6 py-3 bg-mint text-bento-bg font-semibold rounded-lg shadow-md hover:bg-opacity-90 disabled:opacity-50 transition-colors duration-200"
        >
          {creatingTrade ? 'Submitting...' : 'Submit Trade'}
        </button>
        {createTradeError && <p className="text-red-500 mt-4">Error: {createTradeError.message}</p>}
      </div>

      {/* TODO: Display recent trades for the selected portfolio */}
    </div>
  );
};

export default TradePage;
