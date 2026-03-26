## 2024-05-23 - Frontend Worker Explosion
**Learning:** Found that `useWasmPricing` hook was creating a new Web Worker for *every* component instance. In `OptionsChain`, this meant 40+ workers were created on load (one for each cell), causing massive overhead.
**Action:** Implemented a Singleton Worker pattern in the hook. Always check if heavy resources like Workers or WebSocket connections are being created inside hooks used by list items. Use Context or Module-level Singletons to share them.
## 2024-05-24 - WasmGreeksCell Unnecessary Renders
**Learning:** `WasmGreeksCell` is rendered inside a large DataGrid inside `OptionsChain`. Whenever `OptionsChain` state updates (e.g., search or filter changes), all `WasmGreeksCell` instances re-render. Since `WasmGreeksCell` sends messages to the WASM worker, this caused massive overhead.
**Action:** Wrap row components (like `WasmGreeksCell`) that perform expensive async operations (like worker calls) in `React.memo` so they only re-render when their specific props change.

## 2024-05-18 - [Optimize RiskAttributor.aggregate_greeks]
**Learning:** In high-frequency loops dealing with many dictionaries (e.g., parsing positions in a portfolio for risk attribution), avoiding repeated dict lookups inside complex conditional statements (like inline `if` inside `.get()` calls) and using local variables to accumulate values instead of updating dictionary keys yields significant performance improvements (~55% execution time reduction). This is especially impactful for large portfolios due to reduced allocation and conditional operations.
**Action:** Always accumulate running totals in local scope variables during tight loops rather than updating dictionary keys, and flatten conditional branches (e.g., separate logic into option vs linear asset branches) to minimize duplicate dictionary checks within the loop.
