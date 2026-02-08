import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import sys

# Mock strawberry and others
sys.modules["strawberry"] = MagicMock()
sys.modules["strawberry.fastapi"] = MagicMock()

with patch("src.pricing.main.setup_logging"), \
     patch("src.pricing.main.GraphQLRouter"):
    from src.pricing.main import app

class TestPricingMain(unittest.TestCase):
    def test_health(self):
        client = TestClient(app)
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})

if __name__ == '__main__':
    unittest.main()
