import gymnasium as gym
import numpy as np
import structlog

logger = structlog.get_logger()


class TradingEnvironment(gym.Env):
    # ... (existing init)

    def _construct_single_obs(self) -> np.ndarray:
        """Constructs a single 100-dim observation vector with added validation."""
        vec = np.zeros(100, dtype=np.float32)
        # 1. Portfolio state (11 dimensions)
        vec[0] = self.balance / self.initial_balance
        vec[1:11] = self.positions[:10]

        # 2. Market prices (10 dimensions)
        prices = np.array(self.market_data.get("prices", []))
        strikes = np.array(self.market_data.get("strikes", prices))

        # Validation for non-positive prices/strikes before log
        if np.any(prices <= 0) or np.any(strikes <= 0):
            logger.warning(
                "trading_env_non_positive_price_or_strike",
                prices=prices,
                strikes=strikes,
            )
            # Replace with a small positive value or handle as an error condition
            # For now, setting to a default small positive value to avoid NaN
            prices[prices <= 0] = 1e-6
            strikes[strikes <= 0] = 1e-6

        n_prices = min(len(prices), 10)
        if n_prices > 0:
            vec[11 : 11 + n_prices] = np.log(prices[:n_prices] / strikes[:n_prices])

        # 3. Greeks (50 dimensions)
        greeks_raw = self.market_data.get("greeks", [])
        greeks_flat = np.array(greeks_raw).flatten()
        n_greeks = min(len(greeks_flat), 50)
        vec[21 : 21 + n_greeks] = np.tanh(greeks_flat[:n_greeks])

        # 4. Indicators (20 dimensions)
        indicators = self.market_data.get("indicators", [])
        n_ind = min(len(indicators), 20)
        vec[71 : 71 + n_ind] = indicators[:n_ind]

        # Check for NaN/Inf in the final vector
        if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
            logger.error("trading_env_obs_nan_or_inf", obs_vec=vec)
            # Handle by replacing with finite values or raising an error
            vec[np.isnan(vec)] = 0
            vec[np.isinf(vec)] = 0

        return vec

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment with data validation.
        """
        # 0. Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 1. Execute trades (rebalance to target positions)
        trades = action - self.positions

        # 2. Calculate costs
        current_prices = np.array(self.market_data.get("prices", np.zeros(10)))
        if len(current_prices) != 10:
            current_prices = np.pad(current_prices, (0, 10 - len(current_prices)))[:10]

        # Validate current_prices before use
        if np.any(current_prices <= 0):
            logger.warning(
                "trading_env_step_non_positive_prices", prices=current_prices
            )
            # Handle by replacing with a small positive value or triggering truncation
            current_prices[current_prices <= 0] = 1e-6

        transaction_costs = np.sum(
            np.abs(trades) * current_prices * self.transaction_cost
        )
        asset_costs = np.sum(trades * current_prices)

        # 3. Update state
        self.positions = action
        self.balance -= transaction_costs + asset_costs

        # Check for unexpectedly low balance due to errors, beyond normal trading
        if self.balance < -1e9:  # Arbitrary large negative to catch calculation errors
            logger.critical(
                "trading_env_balance_catastrophic_negative", balance=self.balance
            )
            # Force truncation or reset if balance becomes absurdly negative
            return (
                self._get_observation(),
                -100.0,
                True,
                True,
                {},
            )  # Huge penalty, terminate

        # 4. Advance time
        self.current_step += 1
        if self.data_provider:
            self.market_data = self.data_provider.get_data_at_step(self.current_step)
        else:
            self.market_data = self._get_dummy_data()

        # 5. Calculate portfolio value
        # Recalculate current_prices after advancing step as market_data might change
        current_prices = np.array(self.market_data.get("prices", np.zeros(10)))
        if len(current_prices) != 10:
            current_prices = np.pad(current_prices, (0, 10 - len(current_prices)))[:10]

        # Validate current_prices again
        if np.any(current_prices <= 0):
            logger.warning(
                "trading_env_portfolio_non_positive_prices", prices=current_prices
            )
            current_prices[current_prices <= 0] = 1e-6

        option_values = np.sum(self.positions * current_prices)
        portfolio_value = self.balance + option_values
        self.portfolio_values.append(portfolio_value)

        # 6. Calculate reward (Sharpe-like risk adjusted return)
        reward = self._calculate_reward(portfolio_value)

        # 7. Check termination
        terminated = False
        if self.data_provider and self.current_step >= len(self.data_provider) - 1:
            terminated = True
        elif not self.data_provider and self.current_step >= 100:
            terminated = True

        # 8. Check truncation (Drawdown limit)
        truncated = bool(portfolio_value <= self.initial_balance * 0.5)

        info = {
            "portfolio_value": portfolio_value,
            "balance": self.balance,
            "positions": self.positions.copy(),
            "step": self.current_step,
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _calculate_reward(self, current_portfolio_value: float) -> float:
        """
        Multi-objective reward function incorporating risk-adjusted returns with added validation.
        Improved stability using larger window and downside risk focus.
        """
        # 1. Percentage return
        prev_value = self.portfolio_values[-2]
        if prev_value == 0:  # Avoid division by zero
            logger.error(
                "trading_env_reward_div_by_zero",
                prev_value=prev_value,
                current_value=current_portfolio_value,
            )
            return -100.0  # Huge penalty

        ret = (current_portfolio_value - prev_value) / prev_value

        # 2. Volatility penalty (risk adjustment) - Increased window to 20 for stability
        if len(self.portfolio_values) > 20:
            recent_values = np.array(self.portfolio_values[-20:])
            # Validate recent_values before np.diff
            if np.any(recent_values <= 0):
                logger.warning(
                    "trading_env_reward_non_positive_recent_values",
                    values=recent_values,
                )
                recent_values[recent_values <= 0] = 1e-6  # Replace to avoid issues

            recent_returns = np.diff(recent_values) / recent_values[:-1]

            # Check for NaN/Inf in recent_returns
            if np.any(np.isnan(recent_returns)) or np.any(np.isinf(recent_returns)):
                logger.error(
                    "trading_env_reward_returns_nan_or_inf", returns=recent_returns
                )
                volatility = 1.0  # High penalty for instability
            else:
                volatility = np.std(recent_returns)

            vol_penalty = 0.5 * volatility  # Increased weight on risk
        else:
            vol_penalty = 0

        reward = ret - vol_penalty

        # 3. Progressive Drawdown penalty
        drawdown = (
            self.initial_balance - current_portfolio_value
        ) / self.initial_balance
        if drawdown > 0.1:  # 10% threshold
            reward -= 0.1 * (drawdown - 0.1)

        return float(reward)

    def _get_dummy_data(self) -> dict:
        """Generate random data for fallback/tests"""
        return {
            "prices": np.random.uniform(90, 110, 10),
            "greeks": np.random.uniform(-1, 1, (10, 5)),
            "indicators": np.random.uniform(0, 1, 20),
        }
