## 2024-05-23 - Frontend Worker Explosion
**Learning:** Found that `useWasmPricing` hook was creating a new Web Worker for *every* component instance. In `OptionsChain`, this meant 40+ workers were created on load (one for each cell), causing massive overhead.
**Action:** Implemented a Singleton Worker pattern in the hook. Always check if heavy resources like Workers or WebSocket connections are being created inside hooks used by list items. Use Context or Module-level Singletons to share them.
## 2024-05-24 - WasmGreeksCell Unnecessary Renders
**Learning:** `WasmGreeksCell` is rendered inside a large DataGrid inside `OptionsChain`. Whenever `OptionsChain` state updates (e.g., search or filter changes), all `WasmGreeksCell` instances re-render. Since `WasmGreeksCell` sends messages to the WASM worker, this caused massive overhead.
**Action:** Wrap row components (like `WasmGreeksCell`) that perform expensive async operations (like worker calls) in `React.memo` so they only re-render when their specific props change.

## 2024-05-25 - React DataGrid Batch Workers vs Per-Cell Workers
**Learning:** React DataGrid components with deeply nested columns that each perform independent WebWorker calls (e.g. `WasmGreeksCell` calculating options greeks) will bottleneck the frontend by initializing O(N) workers per row render.
**Action:** When working with DataGrids, push heavy async computation to the parent component level (e.g., `OptionsChain`) to take advantage of batch API/Worker methods, then pass the pre-computed results directly to the dumb presentation cell components.

## 2024-05-25 - Dataset Array Enrichment Mapping
**Learning:** Applying UI-level array filters (like a search query `.filter()`) to a dataset *before* matching elements with an externally batch-processed parallel array will break the 1-to-1 index alignment, leading to incorrect data rendering (e.g., wrong option prices applied to the wrong strikes).
**Action:** Always enrich the primary array with batch-processed results using original indices *before* running any destructive array methods like `filter()` or `sort()`.
