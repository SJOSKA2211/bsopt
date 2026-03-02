# TestSprite AI Testing Report

---

## 1️⃣ Document Metadata
- **Project Name:** bsopt
- **Date:** 2026-03-02
- **Prepared by:** TestSprite AI Team
- **Test Mode:** Backend API (Development)

---

## 2️⃣ Requirement Validation Summary

### Requirement: Application Health & Root
#### Test TC001 get root info returns bsopt optimized api message
- **Test Code:** `TC001_get_root_info_returns_bsopt_optimized_api_message.py`
- **Visualization:** [View Dashboard](https://www.testsprite.com/dashboard/mcp/tests/1b82f4fa-170d-42da-befa-658beba29888/ec520486-531e-465f-b07e-8214cb11d8fa)
- **Status:** ✅ Passed
- **Analysis:** The root endpoint correctly returns the API welcome message unauthenticated.

#### Test TC002 get health returns healthy status
- **Test Code:** `TC002_get_health_returns_healthy_status.py`
- **Visualization:** [View Dashboard](https://www.testsprite.com/dashboard/mcp/tests/1b82f4fa-170d-42da-befa-658beba29888/3c2afb2c-9a14-4f6a-b5f4-c3f5ca5688a0)
- **Status:** ✅ Passed
- **Analysis:** The health check endpoint is accessible and correctly reports the service is healthy.

### Requirement: System Observability
#### Test TC003 get metrics returns prometheus metrics text
- **Test Code:** `TC003_get_metrics_returns_prometheus_metrics_text.py`
- **Visualization:** [View Dashboard](https://www.testsprite.com/dashboard/mcp/tests/1b82f4fa-170d-42da-befa-658beba29888/236795e6-351a-43b3-9313-d3202d604d09)
- **Status:** ❌ Failed
- **Analysis:** The `/metrics` endpoint is unexpectedly returning a `401 Unauthorized` status instead of the expected `200`. This suggests that the Prometheus metrics route is inadvertently protected by a global authentication middleware that intercepts the request before it reaches the endpoint.

### Requirement: User Authentication
#### Test TC004 post auth register creates new user with jwt token
- **Test Code:** `TC004_post_auth_register_creates_new_user_with_jwt_token.py`
- **Visualization:** [View Dashboard](https://www.testsprite.com/dashboard/mcp/tests/1b82f4fa-170d-42da-befa-658beba29888/4fe3c73e-a1ab-4224-a53d-5af29380421b)
- **Status:** ❌ Failed
- **Analysis:** The registration endpoint returned a `422 Unprocessable Content` error. This indicates a schema mismatch between the FastAPI Pydantic requirements (missing fields or wrong types in the request body) and what the test sent.

#### Test TC005 post auth login returns access token for valid credentials
- **Test Code:** `TC005_post_auth_login_returns_access_token_for_valid_credentials.py`
- **Visualization:** [View Dashboard](https://www.testsprite.com/dashboard/mcp/tests/1b82f4fa-170d-42da-befa-658beba29888/8037fb62-c680-4a46-819c-0e8814b19a26)
- **Status:** ❌ Failed
- **Analysis:** The login endpoint threw a `500 Internal Server Error`. This points to a severe backend failure during authentication processing, likely related to database connectivity, missing migrations, or a disruption in communication with the Node.js `auth-service`.

#### Test TC006 get auth me returns user profile with valid token
- **Test Code:** `TC006_get_auth_me_returns_user_profile_with_valid_token.py`
- **Status:** ❌ Failed (Cascading Failure)
- **Analysis:** Failed because the test setup requires creating a user via `/register` first, which is currently returning `422`.

### Requirement: Pricing & Options
#### Test TC007 post pricing calculate returns option price and greeks
- **Status:** ❌ Failed (Cascading Failure)
- **Analysis:** Failed during test setup because it requires an authenticated user token. Since the `/login` endpoint is returning `500`, this test could not execute the actual pricing validation.

### Requirement: Portfolio Management
#### Test TC008 get portfolio returns user portfolio
- **Status:** ❌ Failed (Cascading Failure)
- **Analysis:** Failed during test setup due to the `/login` endpoint returning a 500 error.

#### Test TC009 post portfolio positions adds new position
- **Status:** ❌ Failed (Cascading Failure)
- **Analysis:** Failed during test setup due to the `/login` endpoint returning a 500 error.

### Requirement: GraphQL API
#### Test TC010 post graphql returns valid graphql response
- **Status:** ❌ Failed (Cascading Failure)
- **Analysis:** Failed during test setup due to the `/login` endpoint returning a 500 error.

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

1. **Catastrophic Authentication Failure (Critical Risk):**
   - The user login endpoint throws a `500 Internal Server Error`. This implies either the database is unreachable from the API, migrations haven't run preventing table reads, or the Node.js based `auth-service` gateway is failing configuration. Because nearly the entire application relies on authentication tokens, this `500` error cascades and breaks 50% of the entire test suite.

2. **Schema Mismatch on Registration (High Risk):**
   - User registration fails with a `422 Unprocessable Content`. The FastAPI validation layer is rejecting the payload. The API definitions likely require additional mandatory fields (e.g. `full_name` or strict string lengths) that are missing from standard registration flows.

3. **Inappropriate Route Protection (Medium Risk):**
   - The `/metrics` observability endpoint is returning `401 Unauthorized` instead of `200`. The global JWT middleware is intercepting this metrics scrape endpoint. This will break Prometheus monitoring unless the scraper is granted a token (which is anti-pattern) or the route is added to an exclusion list within the middleware.
