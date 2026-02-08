import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock strawberry as a package with submodules
strawberry = MagicMock()
sys.modules["strawberry"] = strawberry
sys.modules["strawberry.fastapi"] = MagicMock()
sys.modules["strawberry.dataloader"] = MagicMock()
sys.modules["strawberry.types"] = MagicMock()
sys.modules["strawberry.federation"] = MagicMock()

# Mock setup_logging to avoid side effects
with patch("src.shared.observability.setup_logging"), \
     patch("src.shared.observability.logging_middleware"), \
     patch("strawberry.fastapi.GraphQLRouter"):
    from src.pricing.main import app

from fastapi.testclient import TestClient

class TestPricingMain(unittest.TestCase):
    def test_health(self):
        client = TestClient(app)
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})

if __name__ == '__main__':
    unittest.main()
