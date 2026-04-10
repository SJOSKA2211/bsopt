## 2026-04-10 - Adding Aria Labels to Icon-only buttons
**Learning:** Icon-only buttons often lack proper accessibility context for screen readers in MUI applications, requiring explicit `aria-label` attributes and optionally `Tooltip` components for mouse users.
**Action:** Always wrap icon-only MUI `IconButton` components in a `Tooltip` and provide a descriptive `aria-label` to ensure full accessibility.
