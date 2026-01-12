import { defineConfig } from 'vite'
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
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './tests/setupTests.ts',
    alias: {
       // Handle CSS imports (modules and regular)
       "\.(css|less|scss|sass)$": 'identity-obj-proxy',
    },
    server: {
      deps: {
        inline: ['@mui/x-data-grid'],
      },
    },
  },
})