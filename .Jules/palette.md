## 2024-05-18 - ARIA labels on IconButtons
**Learning:** Adding ARIA labels to icon-only buttons improves accessibility for screen readers. In OrderTicket.tsx, buttons with only icons (like + and - for quantity/price) lacked context for visually impaired users.
**Action:** Always verify that IconButton components containing solely SVGs/icons include a descriptive aria-label attribute indicating their action (e.g., Increase Quantity).
