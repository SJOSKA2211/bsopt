import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
// import path from 'path' // Unused
import compression from 'vite-plugin-compression'
import fs from 'fs'

// AIOps Heartbeat Plugin: Reports frontend health to the manifold
const AIOpsHeartbeatPlugin = () => ({
  name: 'aiops-heartbeat',
  configureServer(server) {
    const heartbeatPath = '/tmp/frontend_heartbeat';
    const writeHeartbeat = () => {
      const data = {
        time: Date.now() / 1000,
        metrics: {
          health: 'ACTIVE',
          status: 'Vite Dev Server Running',
          processed: 0 // Frontend doesn't "process" ticks in the same way as scrapers
        }
      };
      try {
        fs.writeFileSync(heartbeatPath, JSON.stringify(data));
      } catch (err) {
        console.warn('Failed to write AIOps heartbeat:', err);
      }
    };
    
    // Initial write and interval
    writeHeartbeat();
    const interval = setInterval(writeHeartbeat, 5000);
    
    server.httpServer?.on('close', () => clearInterval(interval));
  }
});

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const isProd = mode === 'production';
  // Use relative base for Netlify/CI previews unless CDN_URL is explicitly set
  const isNetlify = process.env.NETLIFY === 'true';
  const baseUrl = (isProd && !isNetlify) ? process.env.CDN_URL || 'https://cdn.bsopt.com/assets/' : '/';

  return {
    // CDN Setup: Set base path for assets in production
    base: baseUrl,
    assetsInclude: ['**/*.wasm'],
    plugins: [
      react(),
      AIOpsHeartbeatPlugin(),
      compression({
        algorithm: 'brotliCompress',
        ext: '.br',
      }),
      compression({
        algorithm: 'gzip',
        ext: '.gz',
      }),
    ],
    server: {
      host: '0.0.0.0',
      port: 5173,
      proxy: {
        '/api/auth': {
          target: process.env.AUTH_SERVICE_URL || 'http://auth-service:3001',
          changeOrigin: true,
        },
        '/api/v1': {
          target: process.env.API_URL || 'http://api:8000',
          changeOrigin: true,
        },
        '/graphql': {
          target: process.env.GATEWAY_URL || 'http://envoy:8080',
          changeOrigin: true,
        },
        '/health': {
          target: process.env.API_URL || 'http://api:8000',
          changeOrigin: true,
        }
      },
      watch: {
        usePolling: true,
      },
    },
    optimizeDeps: {
      include: ['@apollo/client'],
      exclude: ['bsopt-wasm'] // Prevent Vite from trying to pre-bundle the WASM pkg
    },
    build: {
      sourcemap: true,
      target: 'esnext', // Support top-level await for WASM
      cssCodeSplit: true, // Granular CSS delivery
      assetsInlineLimit: 4096, // Inline small assets to reduce requests
      rollupOptions: {
        output: {
          manualChunks(id) {
            if (id.includes('node_modules')) {
              if (id.includes('@mui') || id.includes('@emotion')) return 'vendor-ui';
              if (id.includes('echarts') || id.includes('lightweight-charts')) return 'vendor-viz';
              if (id.includes('three') || id.includes('@react-three')) return 'vendor-3d';
              if (id.includes('@tanstack/react-query') || id.includes('axios')) return 'vendor-data';
              if (id.includes('react') || id.includes('react-dom') || id.includes('react-router')) return 'vendor-src.shared';
              return 'vendor-utils';
            }
          }
        }
      },
      chunkSizeWarningLimit: 1000,
      minify: 'esbuild',
    },
    test: {
      globals: true,
      environment: 'jsdom',
      setupFiles: './tests/setupTests.ts',
      testTimeout: 15000,
      alias: {
        // Handle CSS imports (modules and regular)
        [/\.(css|less|scss|sass)$/.source]: 'identity-obj-proxy',
      },
      server: {
        deps: {
          inline: ['@mui/x-data-grid'],
        },
      },
    },
  };
});
