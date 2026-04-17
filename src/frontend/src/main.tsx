import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

// Import ApolloProvider and the configured client
import { ApolloProvider } from '@apollo/client';
import { apolloClient } from './lib/apolloClient'; // Assuming client is exported from lib/apolloClient.ts

// Configure the main entry point to use ApolloProvider
const rootElement = document.getElementById('root');
if (rootElement) {
  const root = ReactDOM.createRoot(rootElement);
  root.render(
    <React.StrictMode>
      <ApolloProvider client={apolloClient}>
        <App />
      </ApolloProvider>
    </React.StrictMode>
  );
} else {
  console.error("Root element with ID 'root' not found.");
}
