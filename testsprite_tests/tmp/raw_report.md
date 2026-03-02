
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
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/1b82f4fa-170d-42da-befa-658beba29888/ec520486-531e-465f-b07e-8214cb11d8fa
- **Status:** ✅ Passed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC002 get health returns healthy status
- **Test Code:** [TC002_get_health_returns_healthy_status.py](./TC002_get_health_returns_healthy_status.py)
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/1b82f4fa-170d-42da-befa-658beba29888/3c2afb2c-9a14-4f6a-b5f4-c3f5ca5688a0
- **Status:** ✅ Passed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC003 get metrics returns prometheus metrics text
- **Test Code:** [TC003_get_metrics_returns_prometheus_metrics_text.py](./TC003_get_metrics_returns_prometheus_metrics_text.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 20, in <module>
  File "<string>", line 10, in test_get_metrics_returns_prometheus_metrics_text
AssertionError: Expected status code 200 but got 401

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/1b82f4fa-170d-42da-befa-658beba29888/236795e6-351a-43b3-9313-d3202d604d09
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC004 post auth register creates new user with jwt token
- **Test Code:** [TC004_post_auth_register_creates_new_user_with_jwt_token.py](./TC004_post_auth_register_creates_new_user_with_jwt_token.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 57, in <module>
  File "<string>", line 27, in test_post_auth_register_creates_new_user_with_jwt_token
AssertionError: Expected 201, got 422

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/1b82f4fa-170d-42da-befa-658beba29888/4fe3c73e-a1ab-4224-a53d-5af29380421b
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC005 post auth login returns access token for valid credentials
- **Test Code:** [TC005_post_auth_login_returns_access_token_for_valid_credentials.py](./TC005_post_auth_login_returns_access_token_for_valid_credentials.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 27, in <module>
  File "<string>", line 14, in test_post_auth_login_returns_access_token_for_valid_credentials
AssertionError: Expected status code 200, got 500

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/1b82f4fa-170d-42da-befa-658beba29888/8037fb62-c680-4a46-819c-0e8814b19a26
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC006 get auth me returns user profile with valid token
- **Test Code:** [TC006_get_auth_me_returns_user_profile_with_valid_token.py](./TC006_get_auth_me_returns_user_profile_with_valid_token.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 45, in <module>
  File "<string>", line 23, in test_get_auth_me_returns_user_profile_with_valid_token
  File "/var/task/requests/models.py", line 1024, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 422 Client Error: Unprocessable Content for url: http://127.0.0.1:8000/api/v1/auth/register

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/1b82f4fa-170d-42da-befa-658beba29888/a3a2a214-074c-46ea-9e7a-457f591ed9c8
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC007 post pricing calculate returns option price and greeks
- **Test Code:** [TC007_post_pricing_calculate_returns_option_price_and_greeks.py](./TC007_post_pricing_calculate_returns_option_price_and_greeks.py)
- **Test Error:** Traceback (most recent call last):
  File "<string>", line 15, in test_post_pricing_calculate_returns_option_price_and_greeks
AssertionError: Login failed with status 500

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 54, in <module>
  File "<string>", line 21, in test_post_pricing_calculate_returns_option_price_and_greeks
AssertionError: Authentication step failed: Login failed with status 500

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/1b82f4fa-170d-42da-befa-658beba29888/d0a242ed-3d29-463b-b84f-329bed02ae2e
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC008 get portfolio returns user portfolio
- **Test Code:** [TC008_get_portfolio_returns_user_portfolio.py](./TC008_get_portfolio_returns_user_portfolio.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 36, in <module>
  File "<string>", line 18, in test_get_portfolio_returns_user_portfolio
AssertionError: Login failed with status 500

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/1b82f4fa-170d-42da-befa-658beba29888/97507360-eee8-43b1-ad1b-30d26c269bb4
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC009 post portfolio positions adds new position
- **Test Code:** [TC009_post_portfolio_positions_adds_new_position.py](./TC009_post_portfolio_positions_adds_new_position.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 65, in <module>
  File "<string>", line 18, in test_post_portfolio_positions_adds_new_position
AssertionError: Login failed with status 500

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/1b82f4fa-170d-42da-befa-658beba29888/48869888-19c6-45f3-af99-98f83cf07408
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC010 post graphql returns valid graphql response
- **Test Code:** [TC010_post_graphql_returns_valid_graphql_response.py](./TC010_post_graphql_returns_valid_graphql_response.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 143, in <module>
  File "<string>", line 18, in test_post_graphql_returns_valid_graphql_response
AssertionError: Login failed with status 500

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/1b82f4fa-170d-42da-befa-658beba29888/0ae42c86-f87b-4c2b-80fa-0e1c6779b7d8
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---


## 3️⃣ Coverage & Matching Metrics

- **20.00** of tests passed

| Requirement        | Total Tests | ✅ Passed | ❌ Failed  |
|--------------------|-------------|-----------|------------|
| ...                | ...         | ...       | ...        |
---


## 4️⃣ Key Gaps / Risks
{AI_GNERATED_KET_GAPS_AND_RISKS}
---