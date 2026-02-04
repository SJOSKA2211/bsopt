// src/frontend/src/mocks/browser.ts
import { setupWorker } from 'msw/browser';
import { handlers } from './handlers';

// This configures a Service Worker for the browser
export const worker = setupWorker(...handlers);
