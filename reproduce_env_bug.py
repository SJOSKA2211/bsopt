import numpy as np
from src.ml.reinforcement_learning.trading_env import TradingEnvironment

def test_balance_update():
    env = TradingEnvironment(initial_balance=100000.0, transaction_cost=0.01)
    env.reset()
    
    # Action: buy 10% of portfolio in first asset
    # Action space is Box(-1, 1, shape=(10,))
    action = np.zeros(10)
    action[0] = 0.1
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Initial Balance: 100000.0")
    print(f"Final Balance: {env.balance}")
    print(f"Portfolio Value: {info['portfolio_value']}")
    
    # If the bug is missing asset costs, balance would only decrease by transaction costs.
    # Cost of asset should be approx 10000.
    # Transaction cost should be approx 10000 * 0.01 = 100.
    # Expected balance should be around 89900.
    
    if env.balance > 99000:
        print("BUG DETECTED: Balance did not decrease by asset purchase price!")
    else:
        print("Balance seems to update correctly.")

if __name__ == "__main__":
    test_balance_update()
