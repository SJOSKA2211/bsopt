import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { Workbox } from 'workbox-window';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

// Register service worker for PWA using Workbox
if ('serviceWorker' in navigator && import.meta.env.PROD) {
  const wb = new Workbox('/service-worker.js');

  wb.addEventListener('activated', (event) => {
    console.log('SW activated: ', event);
  });

  wb.register().catch(error => {
    console.error('SW registration failed: ', error);
  });
} else if ('serviceWorker' in navigator && (import.meta.env.DEV || import.meta.env.MODE === 'test')) {
  // Always register in dev/test for verification, even if it doesn't do much without a build
  navigator.serviceWorker.register('/service-worker.js')
    .then(reg => console.log('SW registered (dev/test): ', reg.scope))
    .catch(err => console.log('SW registration failed (dev/test): ', err));
}
