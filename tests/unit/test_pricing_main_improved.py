import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock strawberry as a package with submodules
strawberry = MagicMock()
sys.modules["strawberry"] = strawberry
sys.modules["strawberry.fastapi"] = MagicMock()
sys.modules["strawberry.dataloader"] = MagicMock()
sys.modules["strawberry.types"] = MagicMock()
sys.modules["strawberry.federation"] = MagicMock()

# Mock setup_logging and warmup_jit to avoid side effects
with (
    patch("core.shared.observability.setup_logging"),
    patch("core.shared.observability.logging_middleware"),
    patch("services.quant.pricing.quant_utils.warmup_jit"),
    patch("core.shared.observability.tune_gc"),
    patch("strawberry.fastapi.GraphQLRouter"),
):
    from services.quant.pricing.main import app


class TestPricingMain(unittest.TestCase):
    def test_app_setup(self):
        self.assertEqual(app.title, "BS-Opt Pricing Service")
        # Check if health route exists
        routes = [r.path for r in app.routes]
        self.assertIn("/health", routes)


if __name__ == "__main__":
    unittest.main()
