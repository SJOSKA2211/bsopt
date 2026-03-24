## 2024-05-23 - Frontend Worker Explosion
**Learning:** Found that `useWasmPricing` hook was creating a new Web Worker for *every* component instance. In `OptionsChain`, this meant 40+ workers were created on load (one for each cell), causing massive overhead.
**Action:** Implemented a Singleton Worker pattern in the hook. Always check if heavy resources like Workers or WebSocket connections are being created inside hooks used by list items. Use Context or Module-level Singletons to share them.
## 2024-05-24 - WasmGreeksCell Unnecessary Renders
**Learning:** `WasmGreeksCell` is rendered inside a large DataGrid inside `OptionsChain`. Whenever `OptionsChain` state updates (e.g., search or filter changes), all `WasmGreeksCell` instances re-render. Since `WasmGreeksCell` sends messages to the WASM worker, this caused massive overhead.
**Action:** Wrap row components (like `WasmGreeksCell`) that perform expensive async operations (like worker calls) in `React.memo` so they only re-render when their specific props change.

## $(date +%Y-%m-%d) - RiskAttributor aggregate_greeks performance
**Learning:** In high-frequency loops (e.g., `RiskAttributor.aggregate_greeks` in `src/portfolio/risk.py`), Python dictionary lookups and conditional logic paths can become significant bottlenecks. Pre-accumulating values into local variables (`t_delta`, `t_gamma`, etc.) instead of modifying dictionary values in-place (`totals["delta"]`), and structuring conditional branches cleanly (separating CALL/PUT options from linear assets) significantly speeds up iteration.
**Action:** When working on tight loops handling potentially millions of elements in Python, use local variables to accumulate values and avoid repeated dictionary queries and membership checks inside the loop body.
