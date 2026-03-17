## 2026-03-17 - Added Accessibility Labels to OptionsChain
**Learning:** In MUI, accessibility labels for `TextField` components are applied using the `inputProps` prop (e.g., `inputProps={{ 'aria-label': '...' }}`) rather than directly on the component, whereas `ToggleButtonGroup` components accept `aria-label` directly as a prop.
**Action:** Always ensure `aria-label`s are added correctly based on the MUI component structure to maintain screen reader compatibility.
