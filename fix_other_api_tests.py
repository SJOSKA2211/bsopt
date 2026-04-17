import re
import os

for filename in ["tests/unit/api/test_heston_api.py", "tests/unit/api/test_ml_api.py"]:
    if not os.path.exists(filename): continue
    with open(filename, "r") as f:
        content = f.read()

    fixture = """
from src.auth.auth import get_current_user, get_current_active_user
from src.database.models import User
@pytest.fixture(autouse=True)
def override_auth_v2():
    mock_user = MagicMock(spec=User)
    mock_user.id = "1"
    mock_user.email = "test@example.com"
    mock_user.tier = "pro"
    mock_user.is_active = True
    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    yield
    app.dependency_overrides.clear()
"""
    if "override_auth_v2" not in content:
        content = content.replace("client = TestClient(app, raise_server_exceptions=False)", "client = TestClient(app, raise_server_exceptions=False)\n" + fixture)
        content = content.replace("client = TestClient(app)", "client = TestClient(app)\n" + fixture)
    
    with open(filename, "w") as f:
        f.write(content)

