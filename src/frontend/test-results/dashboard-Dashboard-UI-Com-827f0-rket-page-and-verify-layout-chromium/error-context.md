# Page snapshot

```yaml
- generic [ref=e3]:
  - generic [ref=e4]: "[plugin:vite:import-analysis] Failed to resolve import \"./lib/apollo-client\" from \"src/App.tsx\". Does the file exist?"
  - generic [ref=e5]: /home/kamau/bsopt/src/frontend/src/App.tsx:5:29
  - generic [ref=e6]: "5 | import { BrowserRouter, Routes, Route, useLocation } from \"react-router-dom\"; 6 | import { ApolloProvider } from \"@apollo/client/react\"; 7 | import { apolloClient } from \"./lib/apollo-client\"; | ^ 8 | import { AnimatePresence, motion } from \"framer-motion\"; 9 | import { Layout } from \"./components/layout/Layout\";"
  - generic [ref=e7]: at TransformPluginContext._formatLog (file:///home/kamau/bsopt/src/frontend/node_modules/vite/dist/node/chunks/dep-D4NMHUTW.js:42528:41) at TransformPluginContext.error (file:///home/kamau/bsopt/src/frontend/node_modules/vite/dist/node/chunks/dep-D4NMHUTW.js:42525:16) at normalizeUrl (file:///home/kamau/bsopt/src/frontend/node_modules/vite/dist/node/chunks/dep-D4NMHUTW.js:40504:23) at process.processTicksAndRejections (node:internal/process/task_queues:104:5) at async file:///home/kamau/bsopt/src/frontend/node_modules/vite/dist/node/chunks/dep-D4NMHUTW.js:40623:37 at async Promise.all (index 5) at async TransformPluginContext.transform (file:///home/kamau/bsopt/src/frontend/node_modules/vite/dist/node/chunks/dep-D4NMHUTW.js:40550:7) at async EnvironmentPluginContainer.transform (file:///home/kamau/bsopt/src/frontend/node_modules/vite/dist/node/chunks/dep-D4NMHUTW.js:42323:18) at async loadAndTransform (file:///home/kamau/bsopt/src/frontend/node_modules/vite/dist/node/chunks/dep-D4NMHUTW.js:35739:27) at async viteTransformMiddleware (file:///home/kamau/bsopt/src/frontend/node_modules/vite/dist/node/chunks/dep-D4NMHUTW.js:37254:24
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