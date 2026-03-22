## 2024-03-22 - Add loading spinner to async submit button
**Learning:** Fully replacing a button’s text with a loading spinner removes context and causes subtle layout shifts, reducing user confidence during form submissions. Preserving the text (e.g., "Signing In...") and utilizing the `startIcon` prop with a smaller `CircularProgress` creates a smoother, more accessible loading state.
**Action:** Always maintain contextual text and use the `startIcon` or `endIcon` props on Material UI `Button`s for loading indicators rather than completely swapping out the child text.
