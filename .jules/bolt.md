## 2024-05-23 - Frontend Worker Explosion
**Learning:** Found that `useWasmPricing` hook was creating a new Web Worker for *every* component instance. In `OptionsChain`, this meant 40+ workers were created on load (one for each cell), causing massive overhead.
**Action:** Implemented a Singleton Worker pattern in the hook. Always check if heavy resources like Workers or WebSocket connections are being created inside hooks used by list items. Use Context or Module-level Singletons to share them.
## 2024-05-24 - WasmGreeksCell Unnecessary Renders
**Learning:** `WasmGreeksCell` is rendered inside a large DataGrid inside `OptionsChain`. Whenever `OptionsChain` state updates (e.g., search or filter changes), all `WasmGreeksCell` instances re-render. Since `WasmGreeksCell` sends messages to the WASM worker, this caused massive overhead.
**Action:** Wrap row components (like `WasmGreeksCell`) that perform expensive async operations (like worker calls) in `React.memo` so they only re-render when their specific props change.
## 2024-05-25 - React DataGrid Batch WASM Optimization
**Learning:** Found that rendering individual cell components (like `WasmGreeksCell`) with their own WebWorker calls inside a DataGrid causes massive unnecessary re-renders when parent state updates. Additionally, if the parent batches calculations, it must apply those results back to the *unfiltered* original array before any search filtering is applied, otherwise the 1-to-1 index mappings break (e.g. `enrichedResults[i + half]`).
**Action:** Always map batch-calculated results to the underlying data *before* applying search filters in list/grid components, and pass these pre-calculated results as props to memoized cell components rather than letting each cell perform expensive async calculations.
