## 2024-05-23 - Frontend Worker Explosion
**Learning:** Found that `useWasmPricing` hook was creating a new Web Worker for *every* component instance. In `OptionsChain`, this meant 40+ workers were created on load (one for each cell), causing massive overhead.
**Action:** Implemented a Singleton Worker pattern in the hook. Always check if heavy resources like Workers or WebSocket connections are being created inside hooks used by list items. Use Context or Module-level Singletons to share them.
## 2024-05-24 - WasmGreeksCell Unnecessary Renders
**Learning:** `WasmGreeksCell` is rendered inside a large DataGrid inside `OptionsChain`. Whenever `OptionsChain` state updates (e.g., search or filter changes), all `WasmGreeksCell` instances re-render. Since `WasmGreeksCell` sends messages to the WASM worker, this caused massive overhead.
**Action:** Wrap row components (like `WasmGreeksCell`) that perform expensive async operations (like worker calls) in `React.memo` so they only re-render when their specific props change.
## 2024-05-25 - DataGrid Render Performance (WasmGreeksCell)
**Learning:** Found that `WasmGreeksCell` was still triggering its own `useWasmPricing` hook inside the OptionsChain DataGrid, leading to O(N) WebWorker calls on every render, despite the parent component already fetching batched results.
**Action:** Lifted the state/WASM call up to the parent component (`OptionsChain`). `WasmGreeksCell` now only accepts `price` and `greeks` as props, reducing the number of active WebWorker requests during chain renders to just the batched call, resulting in a significantly faster UI thread.
