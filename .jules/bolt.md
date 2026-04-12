## 2024-05-23 - Frontend Worker Explosion
**Learning:** Found that `useWasmPricing` hook was creating a new Web Worker for *every* component instance. In `OptionsChain`, this meant 40+ workers were created on load (one for each cell), causing massive overhead.
**Action:** Implemented a Singleton Worker pattern in the hook. Always check if heavy resources like Workers or WebSocket connections are being created inside hooks used by list items. Use Context or Module-level Singletons to share them.
## 2024-05-24 - WasmGreeksCell Unnecessary Renders
**Learning:** `WasmGreeksCell` is rendered inside a large DataGrid inside `OptionsChain`. Whenever `OptionsChain` state updates (e.g., search or filter changes), all `WasmGreeksCell` instances re-render. Since `WasmGreeksCell` sends messages to the WASM worker, this caused massive overhead.
**Action:** Wrap row components (like `WasmGreeksCell`) that perform expensive async operations (like worker calls) in `React.memo` so they only re-render when their specific props change.
## 2024-05-25 - React Context vs Zustand Subscriptions
**Learning:** Found that  was subscribing to  directly to get the live price, and passing it down to . Because  updates very frequently via WebSockets, this caused the *entire*  and all of its siblings (which is the entire trading dashboard) to re-render on every single price tick.
**Action:** Move high-frequency Zustand subscriptions as deep into the component tree as possible. Leaf components (like ) should subscribe directly to the store for the specific slice of data they need, rather than relying on props from parent components.
## 2024-05-25 - React Context vs Zustand Subscriptions
**Learning:** Found that MarketPage was subscribing to usePricingStore directly to get the live price, and passing it down to DOMLadder. Because usePricingStore updates very frequently via WebSockets, this caused the *entire* MarketPage and all of its siblings (which is the entire trading dashboard) to re-render on every single price tick.
**Action:** Move high-frequency Zustand subscriptions as deep into the component tree as possible. Leaf components (like DOMLadder) should subscribe directly to the store for the specific slice of data they need, rather than relying on props from parent components.
