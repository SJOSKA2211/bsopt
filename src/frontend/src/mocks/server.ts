// src/frontend/src/mocks/server.ts
import { setupServer } from 'msw/node';
import { handlers } from './handlers';

// This configures a Service Worker for the browser, but an Express-compatible server for Node.js.
// So, it works for both testing and development environments.
export const server = setupServer(...handlers);
