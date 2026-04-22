## 2024-03-24 - Frontend Build Fails on Missing Apollo Client
**Learning:** The project fails `pnpm run build` due to a pre-existing missing module `./lib/apolloClient`. Memory indicates this environment blocker can be safely ignored when making isolated UI/UX changes.
**Action:** Bypass build failure for UI tweaks since vitest suite passes and the issue is known.
