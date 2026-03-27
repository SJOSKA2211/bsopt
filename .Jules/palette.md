## 2024-03-27 - Added ARIA labels to icon buttons in OrderTicket
**Learning:** Found that some icon-only buttons (`IconButton` containing `RemoveIcon` or `AddIcon`) in `OrderTicket` lacked `aria-label` attributes. This is a common accessibility issue for screen readers.
**Action:** Always ensure `IconButton` components that only contain icons have descriptive `aria-label` attributes (e.g., `aria-label="Decrease Quantity"`) to improve accessibility.
