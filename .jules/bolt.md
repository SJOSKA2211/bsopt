## 2024-05-23 - Frontend Worker Explosion
**Learning:** Found that `useWasmPricing` hook was creating a new Web Worker for *every* component instance. In `OptionsChain`, this meant 40+ workers were created on load (one for each cell), causing massive overhead.
**Action:** Implemented a Singleton Worker pattern in the hook. Always check if heavy resources like Workers or WebSocket connections are being created inside hooks used by list items. Use Context or Module-level Singletons to share them.
## 2024-05-24 - WasmGreeksCell Unnecessary Renders
**Learning:** `WasmGreeksCell` is rendered inside a large DataGrid inside `OptionsChain`. Whenever `OptionsChain` state updates (e.g., search or filter changes), all `WasmGreeksCell` instances re-render. Since `WasmGreeksCell` sends messages to the WASM worker, this caused massive overhead.
**Action:** Wrap row components (like `WasmGreeksCell`) that perform expensive async operations (like worker calls) in `React.memo` so they only re-render when their specific props change.
## 2024-05-25 - Python Dictionary Updates in Tight Loops
**Learning:** `aggregate_greeks` in `src/portfolio/risk.py` had a significant performance bottleneck due to repeatedly checking dictionary membership (`"type" in pos`) and accumulating dictionary values inside a tight loop across the portfolio. Memory overhead and lookup times add up.
**Action:** When iterating over large portfolio arrays, accumulate results into local variables (like `t_delta`, `t_gamma`) instead of a dictionary and separate option vs linear asset logic explicitly, removing the need for repeated membership checks. This yields ~35% performance improvement.
