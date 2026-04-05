## 2024-05-23 - Frontend Worker Explosion
**Learning:** Found that `useWasmPricing` hook was creating a new Web Worker for *every* component instance. In `OptionsChain`, this meant 40+ workers were created on load (one for each cell), causing massive overhead.
**Action:** Implemented a Singleton Worker pattern in the hook. Always check if heavy resources like Workers or WebSocket connections are being created inside hooks used by list items. Use Context or Module-level Singletons to share them.
## 2024-05-24 - WasmGreeksCell Unnecessary Renders
**Learning:** `WasmGreeksCell` is rendered inside a large DataGrid inside `OptionsChain`. Whenever `OptionsChain` state updates (e.g., search or filter changes), all `WasmGreeksCell` instances re-render. Since `WasmGreeksCell` sends messages to the WASM worker, this caused massive overhead.
**Action:** Wrap row components (like `WasmGreeksCell`) that perform expensive async operations (like worker calls) in `React.memo` so they only re-render when their specific props change.
## 2024-05-25 - MarketPage Unnecessary Renders
**Learning:** `MarketPage` was directly consuming `usePricingStore` to pass `livePrice` as a prop to `DOMLadder`. Because this store updates very frequently (multiple times per second), the entire `MarketPage` and all its child components (including heavy charts) were re-rendering continuously, causing massive CPU overhead.
**Action:** When using global state like Zustand for high-frequency data, always subscribe to it at the lowest possible component level (e.g., inside `DOMLadder` itself) rather than passing it down from top-level parent pages.
