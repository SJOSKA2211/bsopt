# TestSprite AI Frontend Testing Report

---

## 1️⃣ Document Metadata
- **Project Name:** bsopt
- **Date:** 2026-03-02
- **Prepared by:** TestSprite AI Team
- **Test Mode:** Frontend UI (Vite Server: http://localhost:5173 / API: 8000)

---

## 2️⃣ Requirement Validation Summary

*Note: TestSprite's AI agent generated backend API tests against the frontend server URL instead of pure UI component / Playwright tests, which explains why the frontend test suite output is functionally identical to the backend test suite execution.*

### Requirement: Application Health & Root
#### Test TC001 get root info returns bsopt optimized api message
- **Test Code:** `TC001_get_root_info_returns_bsopt_optimized_api_message.py`
- **Visualization:** [View Dashboard](https://www.testsprite.com/dashboard/mcp/tests/155f0f61-43bb-4557-a4cd-acd60c723575/131c8f6b-afdd-4b73-b394-5ab70627539d)
- **Status:** ✅ Passed

#### Test TC002 get health returns healthy status
- **Test Code:** `TC002_get_health_returns_healthy_status.py`
- **Visualization:** [View Dashboard](https://www.testsprite.com/dashboard/mcp/tests/155f0f61-43bb-4557-a4cd-acd60c723575/3bc56dca-6615-4ac6-a97d-72e5cf4e1417)
- **Status:** ✅ Passed

### Requirement: System Observability
#### Test TC003 get metrics returns prometheus metrics text
- **Visualization:** [View Dashboard](https://www.testsprite.com/dashboard/mcp/tests/155f0f61-43bb-4557-a4cd-acd60c723575/1c6a2e58-e214-45ac-9cc5-9d219b93776a)
- **Status:** ❌ Failed
- **Analysis:** Return status 401 instead of 200, confirming strict auth middleware is blocking Prometheus.

### Requirement: User Authentication
#### Test TC004 post auth register creates new user with jwt token
- **Visualization:** [View Dashboard](https://www.testsprite.com/dashboard/mcp/tests/155f0f61-43bb-4557-a4cd-acd60c723575/bd1fdd40-2e41-4bb2-9c5d-178b306909a0)
- **Status:** ❌ Failed
- **Analysis:** Failed with `422 Unprocessable Content`. The API response explicitly states: `{"detail":[{"type":"missing","loc":["body","password_confirm"],"msg":"Field required"},{"type":"missing","loc":["body","accept_terms"],"msg":"Field required"}]}`. Registration strictly requires `password_confirm` and `accept_terms` fields.

#### Test TC005 post auth login returns access token for valid credentials
- **Visualization:** [View Dashboard](https://www.testsprite.com/dashboard/mcp/tests/155f0f61-43bb-4557-a4cd-acd60c723575/1896471f-2898-444a-89a4-82a998c64dd4)
- **Status:** ❌ Failed
- **Analysis:** Returned `500 Internal Server Error`. The API raised `psycopg2.errors.UndefinedColumn: column users.is_verified does not exist`.

#### Test TC006 get auth me returns user profile with valid token
- **Visualization:** [View Dashboard](https://www.testsprite.com/dashboard/mcp/tests/155f0f61-43bb-4557-a4cd-acd60c723575/62c20af0-f96f-4bff-8ed1-d1bde74f2633)
- **Status:** ❌ Failed

### Requirement: Pricing & Options
#### Test TC007 post pricing calculate returns option price and greeks
- **Status:** ❌ Failed (Cascading Auth Failure)

### Requirement: Portfolio Management
#### Test TC008 get portfolio returns user portfolio
- **Status:** ❌ Failed (Cascading Auth Failure)

#### Test TC009 post portfolio positions adds new position
- **Status:** ❌ Failed (Cascading Auth Failure)

### Requirement: GraphQL API
#### Test TC010 post graphql returns valid graphql response
- **Status:** ❌ Failed (Cascading Auth Failure)

---

## 3️⃣ Coverage & Matching Metrics

- **Success Rate:** 20.00%
- **Total Tests:** 10
- **Passed:** 2
- **Failed:** 8

| Requirement Category | Total Tests | ✅ Passed | ❌ Failed |
|----------------------|-------------|-----------|-----------|
| Application Health & Root | 2 | 2 | 0 |
| System Observability | 1 | 0 | 1 |
| User Authentication | 3 | 0 | 3 |
| Pricing & Options    | 1 | 0 | 1 |
| Portfolio Management | 2 | 0 | 2 |
| GraphQL API          | 1 | 0 | 1 |

---

## 4️⃣ Key Gaps / Risks

1. **Test Plan Overlap (AI Tool Gap):**
   - The TestSprite MCP tool generated standard Python-based API requests (like `/api/v1/auth/login`) instead of Playwright/Puppeteer browser tests for the React frontend, resulting in identical tests to the backend run. TestSprite's backend/frontend detection through MCP needs more explicit guidance in the PRD phase to force UI testing.
2. **Missing Database Migration:**
   - The login endpoint throws a 500 because the `is_verified` column is missing from the `users` table. **A database migration is missing or hasn't been applied to the Postgres container.**
3. **Pydantic Schema Drift:**
   - The user registration endpoint strictly enforces `password_confirm` and `accept_terms` fields, which the standard AI-generated test generator wasn't aware of in the raw OpenAPI schema interpretation.
