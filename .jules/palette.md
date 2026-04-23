## 2024-04-23 - Form Accessibility
**Learning:** Form inputs across the application (like in `market_data.tsx`) often rely on placeholders rather than explicit `<label>` elements, which impacts screen reader accessibility.
**Action:** Always wrap inputs with or provide an explicit `id` and `htmlFor` associated `<label>` element when building or refactoring forms in this codebase.
