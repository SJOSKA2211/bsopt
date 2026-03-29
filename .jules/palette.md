## 2024-05-28 - ARIA Labels for Progress Indicators
**Learning:** Standalone `CircularProgress` and `LinearProgress` components used as loading or progress indicators must include an explicit `aria-label` attribute (e.g., `aria-label="Loading..."`) to prevent `vitest-axe` accessibility violations.
**Action:** Always add an `aria-label` attribute with a descriptive text when using `CircularProgress` or `LinearProgress` components from MUI as standalone loading indicators.
