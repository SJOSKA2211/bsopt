
# TestSprite AI Testing Report(MCP)

---

## 1️⃣ Document Metadata
- **Project Name:** bsopt
- **Date:** 2026-03-02
- **Prepared by:** TestSprite AI Team

---

## 2️⃣ Requirement Validation Summary

#### Test TC001 get root info returns bsopt optimized api message
- **Test Code:** [TC001_get_root_info_returns_bsopt_optimized_api_message.py](./TC001_get_root_info_returns_bsopt_optimized_api_message.py)
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/977363d4-ccaf-4e40-8e46-4fd00deaee45/748f7571-622a-45bc-9441-6e739a962a16
- **Status:** ✅ Passed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC002 get health returns healthy status
- **Test Code:** [TC002_get_health_returns_healthy_status.py](./TC002_get_health_returns_healthy_status.py)
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/977363d4-ccaf-4e40-8e46-4fd00deaee45/ab8d93d1-cfc7-4698-b8e1-95d5ddc20410
- **Status:** ✅ Passed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC003 get metrics returns prometheus metrics text
- **Test Code:** [TC003_get_metrics_returns_prometheus_metrics_text.py](./TC003_get_metrics_returns_prometheus_metrics_text.py)
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/977363d4-ccaf-4e40-8e46-4fd00deaee45/d0b09a02-a241-430f-a41d-2fd70c46416f
- **Status:** ✅ Passed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC004 post auth register creates new user with jwt token
- **Test Code:** [TC004_post_auth_register_creates_new_user_with_jwt_token.py](./TC004_post_auth_register_creates_new_user_with_jwt_token.py)
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/977363d4-ccaf-4e40-8e46-4fd00deaee45/8a3d113f-0096-49f9-8e87-b77149054ca4
- **Status:** ✅ Passed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC005 post auth login returns access token for valid credentials
- **Test Code:** [TC005_post_auth_login_returns_access_token_for_valid_credentials.py](./TC005_post_auth_login_returns_access_token_for_valid_credentials.py)
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/977363d4-ccaf-4e40-8e46-4fd00deaee45/4f08d59e-5568-4afe-b915-ade8c5b3bc80
- **Status:** ✅ Passed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC006 get auth me returns user profile with valid token
- **Test Code:** [TC006_get_auth_me_returns_user_profile_with_valid_token.py](./TC006_get_auth_me_returns_user_profile_with_valid_token.py)
- **Test Error:** Traceback (most recent call last):
  File "<string>", line 34, in test_get_auth_me_returns_user_profile_with_valid_token
  File "/var/task/requests/models.py", line 1024, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url: http://127.0.0.1:8000/api/v1/auth/login

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 62, in <module>
  File "<string>", line 36, in test_get_auth_me_returns_user_profile_with_valid_token
AssertionError: Login failed unexpectedly: 401 Client Error: Unauthorized for url: http://127.0.0.1:8000/api/v1/auth/login

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/977363d4-ccaf-4e40-8e46-4fd00deaee45/707e147a-a783-4690-8b22-fb7f43948401
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC007 post pricing calculate returns option price and greeks
- **Test Code:** [TC007_post_pricing_calculate_returns_option_price_and_greeks.py](./TC007_post_pricing_calculate_returns_option_price_and_greeks.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 54, in <module>
  File "<string>", line 40, in test_post_pricing_calculate_returns_option_price_and_greeks
AssertionError: Expected status 200, got 403

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/977363d4-ccaf-4e40-8e46-4fd00deaee45/9539cb6a-dbc6-4163-9669-7845c0a120b0
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC008 get portfolio returns user portfolio
- **Test Code:** [TC008_get_portfolio_returns_user_portfolio.py](./TC008_get_portfolio_returns_user_portfolio.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 34, in <module>
  File "<string>", line 27, in test_get_portfolio_returns_user_portfolio
AssertionError: Portfolio request failed with status 500

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/977363d4-ccaf-4e40-8e46-4fd00deaee45/14f43b3b-693a-42db-9cc1-f65655c1d077
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC009 post portfolio positions adds new position
- **Test Code:** [TC009_post_portfolio_positions_adds_new_position.py](./TC009_post_portfolio_positions_adds_new_position.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 68, in <module>
  File "<string>", line 19, in test_post_portfolio_positions_adds_new_position
AssertionError: Registration failed: Proxy server error: write EPIPE

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/977363d4-ccaf-4e40-8e46-4fd00deaee45/6806adf0-24e5-4b19-b513-5a126ec109e6
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC010 post graphql returns valid graphql response
- **Test Code:** [TC010_post_graphql_returns_valid_graphql_response.py](./TC010_post_graphql_returns_valid_graphql_response.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 120, in <module>
  File "<string>", line 104, in test_post_graphql_returns_valid_graphql_response
AssertionError: Expected status code 200 but got 500

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/977363d4-ccaf-4e40-8e46-4fd00deaee45/b66fc3f6-5fa6-466c-8fd1-aea057a182c6
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---


## 3️⃣ Coverage & Matching Metrics

- **50.00** of tests passed

| Requirement        | Total Tests | ✅ Passed | ❌ Failed  |
|--------------------|-------------|-----------|------------|
| ...                | ...         | ...       | ...        |
---


## 4️⃣ Key Gaps / Risks
{AI_GNERATED_KET_GAPS_AND_RISKS}
---