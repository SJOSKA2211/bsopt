## 2025-04-24 - Accessibility Labels for Form Inputs
**Learning:** In the `portfolio.tsx` form, text/number inputs lacked explicitly associated `<label>` elements, relying solely on placeholders. This creates an accessibility barrier for screen readers, which require explicit `<label>` tags linked via `htmlFor` matching the input's `id`.
**Action:** Always provide an explicit `id` and `<label htmlFor="id">` for accessibility when building or refactoring forms in this codebase, rather than relying exclusively on placeholders for context.
