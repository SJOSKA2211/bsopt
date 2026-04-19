## 2024-05-19 - Trade Form Accessibility and Feedback
**Learning:** Adding dynamic disabled states with explanatory `title` attributes and `cursor-not-allowed` styles provides immediate, clear feedback to users on complex forms like trade execution, preventing frustration from failed submissions.
**Action:** When implementing forms with required fields, proactively compute a single `isFormInvalid` boolean and apply it alongside visual cues (tooltips and cursors) to guide the user before they attempt submission.
