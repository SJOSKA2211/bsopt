
## 2024-04-02 - Missing ARIA labels on custom numeric input increment/decrement buttons
**Learning:** Custom implementations of numeric inputs in React that rely on icon-only buttons (like MUI's \`IconButton\` with plus/minus icons) frequently lack \`aria-label\` attributes, severely impacting screen reader users' ability to adjust values like order quantities and limit prices.
**Action:** When auditing forms, specifically check custom quantity/price selectors, ensuring that icon-only \`IconButton\`s for increasing/decreasing values have descriptive \`aria-label\`s (e.g., "Decrease quantity").
