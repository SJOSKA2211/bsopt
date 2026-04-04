## 2024-05-23 - Frontend Worker Explosion
**Learning:** Found that `useWasmPricing` hook was creating a new Web Worker for *every* component instance. In `OptionsChain`, this meant 40+ workers were created on load (one for each cell), causing massive overhead.
**Action:** Implemented a Singleton Worker pattern in the hook. Always check if heavy resources like Workers or WebSocket connections are being created inside hooks used by list items. Use Context or Module-level Singletons to share them.
## 2024-05-24 - WasmGreeksCell Unnecessary Renders
**Learning:** `WasmGreeksCell` is rendered inside a large DataGrid inside `OptionsChain`. Whenever `OptionsChain` state updates (e.g., search or filter changes), all `WasmGreeksCell` instances re-render. Since `WasmGreeksCell` sends messages to the WASM worker, this caused massive overhead.
**Action:** Wrap row components (like `WasmGreeksCell`) that perform expensive async operations (like worker calls) in `React.memo` so they only re-render when their specific props change.

## 2024-06-18 - [Optimize Live Price Store Subscription]
**Learning:** High-frequency state updates (like live market prices via WebSocket mapped to Zustand's `usePricingStore`) should never be subscribed to at top-level container components (e.g., `MarketPage`), as this forces continuous React rendering cycles across the entire page hierarchy, leading to severe performance bottlenecks.
**Action:** When working with high-frequency Zustand stores, always move the `useStore` hook call as deep into the component tree as possible (e.g., inside the specific child component like `DOMLadder` that needs it) to isolate re-renders strictly to the nodes that require the data.
