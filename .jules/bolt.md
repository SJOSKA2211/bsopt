## 2024-05-23 - Frontend Worker Explosion
**Learning:** Found that `useWasmPricing` hook was creating a new Web Worker for *every* component instance. In `OptionsChain`, this meant 40+ workers were created on load (one for each cell), causing massive overhead.
**Action:** Implemented a Singleton Worker pattern in the hook. Always check if heavy resources like Workers or WebSocket connections are being created inside hooks used by list items. Use Context or Module-level Singletons to share them.
## 2024-05-24 - WasmGreeksCell Unnecessary Renders
**Learning:** `WasmGreeksCell` is rendered inside a large DataGrid inside `OptionsChain`. Whenever `OptionsChain` state updates (e.g., search or filter changes), all `WasmGreeksCell` instances re-render. Since `WasmGreeksCell` sends messages to the WASM worker, this caused massive overhead.
**Action:** Wrap row components (like `WasmGreeksCell`) that perform expensive async operations (like worker calls) in `React.memo` so they only re-render when their specific props change.

## 2025-02-19 - Pandas Structures in Inner Loops are Anti-Patterns
**Learning:** Heavily relying on Pandas structures (like `pd.Series`, indexing, and concatenation) inside inner loops or recursive algorithms—such as tree traversal in hierarchical clustering (e.g., HRP quasi-diagonalization)—is a severe performance bottleneck. The overhead of instantiating and manipulating these data frames makes operations O(n²), while an equivalent iterative depth-first search (DFS) using standard Python lists runs cleanly in O(n).
**Action:** Always refactor iterative algorithmic tree walks or node extractions inside quantitative calculations to use native Python lists or `numpy` primitives instead of Pandas Series. Pandas should be reserved for top-level vectorized calculations or table alignment, never for inner graph traversal logic.
