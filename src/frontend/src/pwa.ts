import { Workbox } from 'workbox-window';

export const registerServiceWorker = () => {
  if ('serviceWorker' in navigator) {
    if (import.meta.env.PROD) {
      const wb = new Workbox('/service-worker.js');

      wb.addEventListener('activated', (event) => {
        console.log('SW activated: ', event);
      });

      wb.register().catch(error => {
        console.error('SW registration failed: ', error);
      });
    } else if (import.meta.env.DEV || import.meta.env.MODE === 'test') {
      // Always register in dev/test for verification
      navigator.serviceWorker.register('/service-worker.js')
        .then(reg => console.log('SW registered (dev/test): ', reg.scope))
        .catch(err => console.log('SW registration failed (dev/test): ', err));
    }
  }
};
