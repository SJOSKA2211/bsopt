/// <reference types="vitest" />
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'path'
import compression from 'vite-plugin-compression'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    compression({
      algorithm: 'brotliCompress',
      ext: '.br',
    }),
    compression({
      algorithm: 'gzip',
      ext: '.gz',
    }),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules')) {
            if (id.includes('@mui') || id.includes('@emotion')) return 'vendor-ui';
            if (id.includes('echarts') || id.includes('lightweight-charts')) return 'vendor-viz';
            if (id.includes('three') || id.includes('@react-three')) return 'vendor-3d';
            if (id.includes('@tanstack/react-query') || id.includes('axios')) return 'vendor-data';
            if (id.includes('react') || id.includes('react-dom') || id.includes('react-router')) return 'vendor-core';
            return 'vendor-utils';
          }
        }
      }
    },
    chunkSizeWarningLimit: 1000,
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './tests/setupTests.ts',
    alias: {
       // Handle CSS imports (modules and regular)
       "\\.(css|less|scss|sass)$": 'identity-obj-proxy',
    },
    server: {
      deps: {
        inline: ['@mui/x-data-grid'],
      },
    },
  },
})