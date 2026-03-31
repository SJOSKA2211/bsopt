## 2024-03-31 - Add Missing aria-labels to Progress bars and IconButtons
**Learning:** Found several MUI `CircularProgress`, `LinearProgress`, and `IconButton` instances missing `aria-label`s, which led to vitest-axe failures. Standalone progress components and icon-only buttons strictly require `aria-label` attributes for screen reader accessibility in this application.
**Action:** Always ensure any progress bar or icon-only button includes a descriptive `aria-label` attribute when creating or modifying them in the frontend codebase.
