## 2025-03-30 - Add aria-label to Progress Components
**Learning:** Standalone `CircularProgress` and `LinearProgress` components used as loading or progress indicators can cause accessibility violations (vitest-axe) if they lack an explicit `aria-label` attribute describing their purpose.
**Action:** Always include an `aria-label` (e.g., `aria-label="Loading..."` or dynamic labels like `aria-label={`${label} progress`}`) when implementing progress indicators.
