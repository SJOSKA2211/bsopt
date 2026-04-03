## 2024-05-18 - Tooltip Wrapped Icon-Only Buttons
**Learning:** Tooltip components (like MUI's `<Tooltip>`) do not inherently provide accessible names to the interactive elements they wrap. Icon-only buttons wrapped in Tooltips still require an explicit `aria-label` attribute to be accessible to screen readers.
**Action:** Always ensure an `aria-label` is applied directly to `<IconButton>` or `<button>` components containing only icons, even if a Tooltip visually explains their purpose.
