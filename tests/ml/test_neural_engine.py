import numpy as np

from src.math_kernel.factory import PricingEngineFactory
from src.math_kernel.models import BSParameters


def test_neural_engine_lifecycle():
    # 1. Get Engine (lazy load)
    # Note: factory returns BasePricingEngine, but we know it's Neural
    engine = PricingEngineFactory.get_engine("neural")

    # 2. Train (Dummy Data)
    # Inputs: [Spot, Strike, T, Vol, R, Q]
    X = np.random.rand(10, 6).astype(np.float32)
    # Ensure reasonable ranges for stability
    X[:, 0] = X[:, 0] * 100 + 50  # Spot
    X[:, 1] = X[:, 1] * 100 + 50  # Strike
    X[:, 2] = X[:, 2] + 0.1  # T
    X[:, 3] = 0.2  # Vol
    X[:, 4] = 0.05  # R
    X[:, 5] = 0.0  # Q

    y = np.random.rand(10, 1).astype(np.float32) * 10

    print("Training...")
    # Dynamic dispatch to specific method
    engine.train_model(X, y, epochs=1)

    # 3. Inference
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.0
    )
    price = engine.price(params)
    print(f"Price: {price}")
    assert isinstance(price, float)

    # 4. Greeks
    greeks = engine.calculate_greeks(params)
    print(f"Greeks: {greeks}")
    assert greeks.delta is not None
    assert isinstance(greeks.gamma, float)


if __name__ == "__main__":
    test_neural_engine_lifecycle()
