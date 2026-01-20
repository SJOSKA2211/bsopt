/// <reference types="vitest" />
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
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
            if (id.includes('@mui') || id.includes('@emotion')) return 'vendor-mui';
            if (id.includes('echarts') || id.includes('lightweight-charts')) return 'vendor-charts';
            if (id.includes('three') || id.includes('@react-three')) return 'vendor-three';
            if (id.includes('@tanstack/react-query')) return 'vendor-query';
            return 'vendor';
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