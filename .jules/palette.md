
## 2025-02-14 - Add aria-label to Icon-Only Buttons
**Learning:** Icon-only buttons (like `IconButton` with `RemoveIcon` or `AddIcon`) used frequently in forms and controls for modifying quantitative inputs (e.g., trading order tickets) lack accessible names by default, rendering them unreadable by screen readers.
**Action:** Always verify that every icon-only button includes an explicit `aria-label` attribute (e.g., `aria-label="Decrease quantity"`) to ensure the application remains fully accessible.
