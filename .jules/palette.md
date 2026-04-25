## 2025-04-25 - Form Labels in React/Tailwind Components
**Learning:** Found form inputs in custom components (like portfolios) missing explicit label tags and relying solely on placeholders. This is an accessibility issue.
**Action:** When building or refactoring forms, always associate a clear `<label>` using the `htmlFor` attribute that matches the `id` of the `<input>` element.
