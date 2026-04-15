## 2024-04-15 - Material UI IconButton Accessibility
**Learning:** Icon-only IconButton components in Material UI need both an explicit `aria-label` for screen reader users and a wrapper `Tooltip` for mouse users to provide context.
**Action:** Always wrap IconButtons in a Tooltip and ensure they have a matching `aria-label`.
