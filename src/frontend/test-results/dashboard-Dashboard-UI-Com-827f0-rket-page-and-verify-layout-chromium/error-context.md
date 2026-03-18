# Page snapshot

```yaml
- generic [ref=e3]:
  - generic [ref=e4]: "[plugin:vite:import-analysis] Failed to resolve import \"./lib/apollo-client\" from \"src/App.tsx\". Does the file exist?"
  - generic [ref=e5]: /home/kamau/bsopt/src/frontend/src/App.tsx:7:29
  - generic [ref=e6]: "5 | import { BrowserRouter, Routes, Route } from \"react-router-dom\"; 6 | import { ApolloProvider } from \"@apollo/client/react\"; 7 | import { apolloClient } from \"./lib/apollo-client\"; | ^ 8 | import { Box, CircularProgress } from \"@mui/material\"; 9 | import { AnimatePresence, motion } from \"framer-motion\";"
  - generic [ref=e7]: at TransformPluginContext._formatLog (file:///home/kamau/bsopt/src/frontend/node_modules/.pnpm/rolldown-vite@7.2.5_@types+node@24.10.7_esbuild@0.25.12/node_modules/rolldown-vite/dist/node/chunks/node.js:30679:43) at TransformPluginContext.error (file:///home/kamau/bsopt/src/frontend/node_modules/.pnpm/rolldown-vite@7.2.5_@types+node@24.10.7_esbuild@0.25.12/node_modules/rolldown-vite/dist/node/chunks/node.js:30676:14) at normalizeUrl (file:///home/kamau/bsopt/src/frontend/node_modules/.pnpm/rolldown-vite@7.2.5_@types+node@24.10.7_esbuild@0.25.12/node_modules/rolldown-vite/dist/node/chunks/node.js:28717:18) at async file:///home/kamau/bsopt/src/frontend/node_modules/.pnpm/rolldown-vite@7.2.5_@types+node@24.10.7_esbuild@0.25.12/node_modules/rolldown-vite/dist/node/chunks/node.js:28780:32 at async Promise.all (index 6) at async TransformPluginContext.transform (file:///home/kamau/bsopt/src/frontend/node_modules/.pnpm/rolldown-vite@7.2.5_@types+node@24.10.7_esbuild@0.25.12/node_modules/rolldown-vite/dist/node/chunks/node.js:28748:4) at async EnvironmentPluginContainer.transform (file:///home/kamau/bsopt/src/frontend/node_modules/.pnpm/rolldown-vite@7.2.5_@types+node@24.10.7_esbuild@0.25.12/node_modules/rolldown-vite/dist/node/chunks/node.js:30468:14) at async loadAndTransform (file:///home/kamau/bsopt/src/frontend/node_modules/.pnpm/rolldown-vite@7.2.5_@types+node@24.10.7_esbuild@0.25.12/node_modules/rolldown-vite/dist/node/chunks/node.js:21586:26)
  - generic [ref=e8]:
    - text: Click outside, press Esc key, or fix the code to dismiss.
    - text: You can also disable this overlay by setting
    - code [ref=e9]: server.hmr.overlay
    - text: to
    - code [ref=e10]: "false"
    - text: in
    - code [ref=e11]: vite.config.ts
    - text: .
```