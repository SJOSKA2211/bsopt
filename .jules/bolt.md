## 2024-05-23 - Frontend Worker Explosion
**Learning:** Found that `useWasmPricing` hook was creating a new Web Worker for *every* component instance. In `OptionsChain`, this meant 40+ workers were created on load (one for each cell), causing massive overhead.
**Action:** Implemented a Singleton Worker pattern in the hook. Always check if heavy resources like Workers or WebSocket connections are being created inside hooks used by list items. Use Context or Module-level Singletons to share them.
## 2024-05-24 - WasmGreeksCell Unnecessary Renders
**Learning:** `WasmGreeksCell` is rendered inside a large DataGrid inside `OptionsChain`. Whenever `OptionsChain` state updates (e.g., search or filter changes), all `WasmGreeksCell` instances re-render. Since `WasmGreeksCell` sends messages to the WASM worker, this caused massive overhead.
**Action:** Wrap row components (like `WasmGreeksCell`) that perform expensive async operations (like worker calls) in `React.memo` so they only re-render when their specific props change.

## 2024-11-20 - OptionsChain Greeks Cell Renders
**Learning:** `WasmGreeksCell` instances were making separate Web Worker calls inside a large `OptionsChain` DataGrid. Even though the hooks used a Singleton Worker pattern, `OptionsChain` was already computing batch enrichment for the entire chain. Having individual cells trigger redundant hook logic and async state caused significant render overhead and duplicated data flows.
**Action:** Lift the WASM calculation state up to the parent list component (`OptionsChain`) by adding `vega`, `theta`, and `rho` to the `batchCalculate` outputs, and apply the enrichment before mapping to rows. Pass the resulting `price` and `greeks` directly as props to `React.memo` cell components (`WasmGreeksCell`) to ensure clean, prop-driven renders without triggering internal hook state or duplicate async operations.
