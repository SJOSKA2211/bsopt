import sys

sys.path.append(".")

from src.pricing.black_scholes import BlackScholesEngine
from src.pricing.models import BSParameters

params = BSParameters(
    spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.02
)

print("Starting pricing test...")
try:
    price = BlackScholesEngine.price_call(params)
    print(f"Price: {price}")
except Exception:
    import traceback

    traceback.print_exc()
    sys.exit(1)
