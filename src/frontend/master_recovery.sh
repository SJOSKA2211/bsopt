#!/bin/bash
ROOT="/home/kamau/bsopt/src/frontend"

force_write() {
  FILE="$1"
  CONTENT="$2"
  rm -f "$FILE"
  echo "$CONTENT" > "$FILE"
}

# 1. package.json
rm -f "$ROOT/package.json"
cat > "$ROOT/package.json" <<'EOF'
{
  "name": "@bsopt/frontend",
  "private": true,
  "version": "6.4.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build"
  },
  "dependencies": {
    "@apollo/client": "^3.12.0",
    "@emotion/react": "^11.14.0",
    "@emotion/styled": "^11.14.0",
    "@mui/icons-material": "^6.3.0",
    "@mui/material": "^6.3.0",
    "@tanstack/react-query": "^5.62.0",
    "framer-motion": "^11.15.0",
    "graphql": "^16.10.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^7.1.0",
    "zustand": "^5.0.2"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.4",
    "autoprefixer": "^10.4.20",
    "postcss": "^8.4.49",
    "tailwindcss": "^3.4.16",
    "typescript": "5.7.2",
    "vite": "^6.0.3"
  }
}
EOF

# 2. tsconfigs
rm -f "$ROOT/tsconfig.json"
cat > "$ROOT/tsconfig.json" <<'EOF'
{
  "files": [],
  "references": [
    { "path": "./tsconfig.app.json" },
    { "path": "./tsconfig.node.json" }
  ]
}
EOF

rm -f "$ROOT/tsconfig.app.json"
cat > "$ROOT/tsconfig.app.json" <<'EOF'
{
  "compilerOptions": {
    "composite": true,
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "strict": true,
    "baseUrl": ".",
    "paths": { "@/*": ["./src/*"] }
  },
  "include": ["src"]
}
EOF

rm -f "$ROOT/tsconfig.node.json"
cat > "$ROOT/tsconfig.node.json" <<'EOF'
{
  "compilerOptions": {
    "composite": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true
  },
  "include": ["vite.config.ts"]
}
EOF

# 3. vite.config.ts
rm -f "$ROOT/vite.config.ts"
cat > "$ROOT/vite.config.ts" <<'EOF'
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { '@': path.resolve(__dirname, './src') } },
  server: { port: 5175, host: '0.0.0.0' }
});
EOF

# 4. Source Files (Minified Premium)
mkdir -p "$ROOT/src/components/layout"
mkdir -p "$ROOT/src/pages/dashboard"

rm -f "$ROOT/src/index.css"
cat > "$ROOT/src/index.css" <<'EOF'
@tailwind base; @tailwind components; @tailwind utilities;
:root { --color-bento-bg: #010409; --color-mint: #00ffa3; }
body { background: var(--color-bento-bg); color: #fff; margin: 0; font-family: sans-serif; }
.bento-grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 1.5rem; width: 100%; }
.bento-card { background: rgba(13,17,23,0.75); border: 1px solid rgba(48,54,61,0.4); border-radius: 16px; padding: 1.5rem; backdrop-filter: blur(12px); }
.label-secondary { font-size: 10px; font-weight: 900; text-transform: uppercase; color: rgba(255,255,255,0.4); }
EOF

rm -f "$ROOT/src/App.tsx"
cat > "$ROOT/src/App.tsx" <<'EOF'
import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/layout/Layout';
import DashboardPage from './pages/dashboard/DashboardPage';
const App = () => (
  <BrowserRouter>
    <Layout>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<DashboardPage />} />
      </Routes>
    </Layout>
  </BrowserRouter>
);
export default App;
EOF

rm -f "$ROOT/src/components/layout/Layout.tsx"
cat > "$ROOT/src/components/layout/Layout.tsx" <<'EOF'
import React from 'react';
export const Layout = ({ children }: any) => (
  <div className="flex h-screen bg-bento-bg text-white overflow-hidden">
     <aside className="w-[280px] border-r border-white/5 p-8">
        <h1 className="text-2xl font-black text-mint">BS-OPT</h1>
     </aside>
     <div className="flex flex-col flex-grow h-screen overflow-hidden">
        <header className="h-16 border-b border-white/5 flex items-center px-8">
           <span className="text-[10px] font-black opacity-30">TERMINAL_v6.4</span>
        </header>
        <main className="flex-grow overflow-auto p-8">{children}</main>
     </div>
  </div>
);
EOF

rm -f "$ROOT/src/pages/dashboard/DashboardPage.tsx"
cat > "$ROOT/src/pages/dashboard/DashboardPage.tsx" <<'EOF'
import React from 'react';
const DashboardPage = () => (
  <div className="bento-grid">
     <div className="col-span-12 lg:col-span-4 bento-card">
        <span className="label-secondary">NET_LIQUIDATION</span>
        <div className="text-3xl font-black mt-2 font-mono">$254,120.42</div>
     </div>
     <div className="col-span-12 lg:col-span-8 bento-card h-[400px]">
        <span className="label-secondary">SIGNAL_ENGINE</span>
     </div>
  </div>
);
export default DashboardPage;
EOF

rm -f "$ROOT/src/main.tsx"
cat > "$ROOT/src/main.tsx" <<'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';
ReactDOM.createRoot(document.getElementById('root')!).render(<App />);
EOF

echo "MASTER RECOVERY COMPLETE."
