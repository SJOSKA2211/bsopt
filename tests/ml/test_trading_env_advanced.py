import numpy as np
from src.ml.reinforcement_learning.trading_env import TradingEnvironment

def test_reward_function_risk_awareness():
    """Test that the reward function accounts for volatility/risk."""
    env = TradingEnvironment(initial_balance=100000)
    
    # Low volatility scenario: small movements, final step 102k -> 103k
    env.portfolio_values = [100000, 100500, 101000, 101500, 102000, 103000]
    reward_low_vol = env._calculate_reward(103000)
    
    # High volatility scenario: large movements, but SAME final step 102k -> 103k
    env.portfolio_values = [100000, 150000, 50000, 150000, 102000, 103000]
    reward_high_vol = env._calculate_reward(103000)
    
    # Both have same 'ret' in the last step: (103000 - 102000) / 102000
    # But high_vol has much higher historical volatility in the last 5 steps.
    assert reward_high_vol < reward_low_vol

def test_transaction_cost_deduction():
    """Test that transaction costs are accurately deducted from balance."""
    env = TradingEnvironment(initial_balance=100000, transaction_cost=0.01) # 1% cost
    env.reset()
    
    # Current state: balance=100000, positions=[0...0]
    # Price is 100 for all options (dummy data)
    env.market_data = {'prices': np.ones(10) * 100.0}
    
    # Action: Buy 1 unit of each option. Total cost = 10 * 100 = 1000.
    # Transaction cost = 0.01 * 1000 = 10.
    action = np.ones(10)
    env.step(action)
    
    # New balance should be 100000 - 10 = 99990.
    # Wait, the current implementation does: 
    # self.balance -= transaction_costs
    # It DOES NOT deduct the purchase price of the options?
    # Usually in these environments, 'balance' is cash. buying an option reduces cash and increases position value.
    # Let's check the code:
    # trades = action - self.positions
    # transaction_costs = np.sum(np.abs(trades) * current_prices * self.transaction_cost)
    # self.positions = action
    # self.balance -= transaction_costs
    
    # BUG IDENTIFIED: The implementation only subtracts transaction_costs, 
    # but doesn't subtract the actual cost of buying the asset from the cash balance.
    assert env.balance == 100000 - 10 - 1000 # Correct behavior: cash decreases by (cost + asset_price)

def test_action_clipping():
    """Test that actions are clipped to the action space."""
    env = TradingEnvironment()
    env.reset()
    
    # Action space is -1 to 1.
    action = np.ones(10) * 5.0 # Way out of bounds
    env.step(action)
    
    # Environment should clip the action or handle it.
    assert np.all(env.positions <= 1.0)
    assert np.all(env.positions >= -1.0)
