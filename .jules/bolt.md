## 2024-05-23 - Frontend Worker Explosion
**Learning:** Found that `useWasmPricing` hook was creating a new Web Worker for *every* component instance. In `OptionsChain`, this meant 40+ workers were created on load (one for each cell), causing massive overhead.
**Action:** Implemented a Singleton Worker pattern in the hook. Always check if heavy resources like Workers or WebSocket connections are being created inside hooks used by list items. Use Context or Module-level Singletons to share them.
## 2024-05-24 - WasmGreeksCell Unnecessary Renders
**Learning:** `WasmGreeksCell` is rendered inside a large DataGrid inside `OptionsChain`. Whenever `OptionsChain` state updates (e.g., search or filter changes), all `WasmGreeksCell` instances re-render. Since `WasmGreeksCell` sends messages to the WASM worker, this caused massive overhead.
**Action:** Wrap row components (like `WasmGreeksCell`) that perform expensive async operations (like worker calls) in `React.memo` so they only re-render when their specific props change.
## 2024-05-25 - React.memo and Zustand state subscription optimizations
**Learning:** Found that `MarketPage` was passing `currentPrice` down to `DOMLadder` by subscribing to the `usePricingStore`. This caused the entire `MarketPage` and all its components (like charting components) to re-render constantly when high-frequency price updates occurred.
**Action:** Subscribed the localized state (`currentPrice` via `usePricingStore`) directly within `DOMLadder` to isolate high-frequency updates and avoid passing it as a prop from a top-level parent page.

## 2024-05-25 - useCallback dependencies and refs
**Learning:** Encountered `useWebSocket` hook reconnection bug where the `connect` callback, inside an event handler, referenced an outdated version of itself or triggered "Cannot access variable before it is declared" errors due to circular dependencies and Temporal Dead Zone limits in React Hooks.
**Action:** Use a mutable React `useRef` to store the latest version of the `connect` callback (e.g. `connectRef.current = connect`), and call it via the ref inside asynchronous callbacks (like `setTimeout`) to sidestep stale closures and TDZ issues without adding `connect` as a dependency.
