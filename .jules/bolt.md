## 2024-05-23 - Frontend Worker Explosion
**Learning:** Found that `useWasmPricing` hook was creating a new Web Worker for *every* component instance. In `OptionsChain`, this meant 40+ workers were created on load (one for each cell), causing massive overhead.
**Action:** Implemented a Singleton Worker pattern in the hook. Always check if heavy resources like Workers or WebSocket connections are being created inside hooks used by list items. Use Context or Module-level Singletons to share them.
## 2024-05-24 - WasmGreeksCell Unnecessary Renders
**Learning:** `WasmGreeksCell` is rendered inside a large DataGrid inside `OptionsChain`. Whenever `OptionsChain` state updates (e.g., search or filter changes), all `WasmGreeksCell` instances re-render. Since `WasmGreeksCell` sends messages to the WASM worker, this caused massive overhead.
**Action:** Wrap row components (like `WasmGreeksCell`) that perform expensive async operations (like worker calls) in `React.memo` so they only re-render when their specific props change.
## 2024-05-24 - Numba JIT vs NumPy cumsum
**Learning:** Pure NumPy operations using cumulative sums for sliding windows (e.g., `cumsum`, `cumsum_sq`) allocate extremely large arrays and cause high memory pressure/GC overhead on large arrays (1M+ elements).
**Action:** Replace `cumsum` sliding windows with a Numba `@njit(fastmath=True)` kernel that performs an $O(1)$ running sum/variance update loop. This yields a substantial 3x-9x speedup by completely avoiding large intermediary array allocations.
