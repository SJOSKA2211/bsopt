## 2024-05-23 - Frontend Worker Explosion
**Learning:** Found that `useWasmPricing` hook was creating a new Web Worker for *every* component instance. In `OptionsChain`, this meant 40+ workers were created on load (one for each cell), causing massive overhead.
**Action:** Implemented a Singleton Worker pattern in the hook. Always check if heavy resources like Workers or WebSocket connections are being created inside hooks used by list items. Use Context or Module-level Singletons to share them.
## 2024-05-24 - WasmGreeksCell Unnecessary Renders
**Learning:** `WasmGreeksCell` is rendered inside a large DataGrid inside `OptionsChain`. Whenever `OptionsChain` state updates (e.g., search or filter changes), all `WasmGreeksCell` instances re-render. Since `WasmGreeksCell` sends messages to the WASM worker, this caused massive overhead.
**Action:** Wrap row components (like `WasmGreeksCell`) that perform expensive async operations (like worker calls) in `React.memo` so they only re-render when their specific props change.
## 2026-04-02 - [Risk Attributor Loop Optimization]
**Learning:** In high-frequency Python loops calculating risk exposures over large portfolios, repetitive dictionary lookups and conditional checks within the loop are slow. Specifically, checking `"type" in pos` and accessing dictionaries iteratively caused a measurable bottleneck in `RiskAttributor.aggregate_greeks`.
**Action:** In loops over large dict collections, accumulate totals in fast local variables (rather than updating a dictionary) and separate complex conditional logic (e.g., options vs linear assets) to minimize repeated lookups.
