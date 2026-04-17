import re
import os

for filename in ["tests/unit/api/test_pricing_api.py", "tests/unit/api/test_heston_api.py", "tests/unit/api/test_ml_api.py"]:
    if not os.path.exists(filename): continue
    with open(filename, "r") as f:
        content = f.read()

    # Add override_auth fixture
    if "override_auth" not in content:
        fixture = """
from api.middleware.jwt_validator import require_auth
@pytest.fixture(autouse=True)
def override_auth():
    mock_claims = MagicMock()
    mock_claims.tier = "pro"
    app.dependency_overrides[require_auth] = lambda: mock_claims
    yield
    app.dependency_overrides.clear()
"""
        content = content.replace("client = TestClient(app, raise_server_exceptions=False)", "client = TestClient(app, raise_server_exceptions=False)\n" + fixture)
        content = content.replace("client = TestClient(app)", "client = TestClient(app)\n" + fixture)
        
    # Replace get_strategy with get_engine
    content = content.replace('patch("src.math_kernel.factory.PricingEngineFactory.get_strategy"', 'patch("src.math_kernel.factory.PricingEngineFactory.get_engine"')
    
    with open(filename, "w") as f:
        f.write(content)

