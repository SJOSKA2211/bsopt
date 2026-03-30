## 2024-05-23 - Frontend Worker Explosion
**Learning:** Found that `useWasmPricing` hook was creating a new Web Worker for *every* component instance. In `OptionsChain`, this meant 40+ workers were created on load (one for each cell), causing massive overhead.
**Action:** Implemented a Singleton Worker pattern in the hook. Always check if heavy resources like Workers or WebSocket connections are being created inside hooks used by list items. Use Context or Module-level Singletons to share them.
## 2024-05-24 - WasmGreeksCell Unnecessary Renders
**Learning:** `WasmGreeksCell` is rendered inside a large DataGrid inside `OptionsChain`. Whenever `OptionsChain` state updates (e.g., search or filter changes), all `WasmGreeksCell` instances re-render. Since `WasmGreeksCell` sends messages to the WASM worker, this caused massive overhead.
**Action:** Wrap row components (like `WasmGreeksCell`) that perform expensive async operations (like worker calls) in `React.memo` so they only re-render when their specific props change.

## 2024-03-30 - Optimize Greeks Aggregation

**Learning:** In high-frequency loops (like aggregating portfolio greeks across thousands of positions in `RiskAttributor.aggregate_greeks`), accumulating values in local variables and separating conditional logic branches (e.g., options vs linear assets) minimizes dictionary lookups and membership checks within the iteration, yielding ~50% faster execution. Adding an early return (or continue) for 0 quantity positions skips unnecessary calculations.

**Action:** Accumulate into local variables inside hot loops instead of dicts, unroll complex inline dict `.get()` checks with conditional defaults into simple if/else branching, and skip work early for zero-values.
