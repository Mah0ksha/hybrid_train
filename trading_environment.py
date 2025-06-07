import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, List, Any

from risk_config import RiskConfig
from risk_manager import RiskManager, RiskManagerEnvInterface
from logger_setup import setup_logger
from config_vals import START_CAP, TRANSACTION_COST, MAX_DRAWDOWN, RISK_PER_TRADE, STOP_LOSS, MAX_POSITION_SIZE, TRAILING_STOP_PERCENT, REWARD_SCALE
from trading_metrics import EnhancedTradingMetrics

logger = setup_logger()

class TradingEnv(gym.Env, RiskManagerEnvInterface):
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, risk_manager: Optional[RiskManager] = None, window_size: int = 20, max_steps: int = 1000, debug: bool = False, reward_scale: float = 1.0, stop_loss_penalty: float = 1.0, holding_penalty: float = 0.001, profit_incentive: float = 0.1, novelty_reward_scale: float = 0.0001):
        super().__init__()
        self.df = df.copy()
        self.symbol = self.df['symbol'].iloc[0] if 'symbol' in self.df.columns else 'UNKNOWN'
        self.window_size = window_size
        self.max_steps = max_steps
        self.debug = debug
        self.reward_scale = reward_scale
        self.stop_loss_penalty = stop_loss_penalty
        self.holding_penalty = holding_penalty
        self.profit_incentive = profit_incentive
        self.returns = []
        
        self.prices = self.df['close'].to_numpy()
        self.timestamps = pd.to_datetime(self.df.index).values if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(pd.Series(self.df.index)).values
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, df.shape[1]), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell

        self.visited_states = set()
        self.novelty_reward_scale = novelty_reward_scale

        self.risk_manager = risk_manager if risk_manager else self._create_default_risk_manager()
        self.risk_manager.env_interface = self
        
        self._balance = START_CAP
        self.reset()
        logger.info(f"Trading environment initialized for symbol: {self.symbol}")

    def _create_default_risk_manager(self) -> RiskManager:
        env_risk_config = RiskConfig(
            base_stop_loss=STOP_LOSS,
            base_take_profit=0.04,
            max_position_size=MAX_POSITION_SIZE,
            risk_per_trade=RISK_PER_TRADE,
            transaction_cost=TRANSACTION_COST,
            max_portfolio_risk=MAX_DRAWDOWN
        )
        return RiskManager(config=env_risk_config, env_interface=self, initial_capital=START_CAP)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = self.window_size
        self._position = 0.0
        self.entry_price = 0.0
        self._balance = START_CAP
        self.portfolio_value = START_CAP
        self.last_trade_step = 0
        self.episode_trade_details = []
        self.current_trade_open_details = None
        self.rm_closed_trade_this_step = False
        self.visited_states.clear()
        if self.risk_manager:
            self.risk_manager.reset()
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.rm_closed_trade_this_step = False
        current_price = self._get_current_price()
        current_timestamp = self.timestamps[self.current_step]
        closure_reason = None
        
        if self._position != 0:
            closed_by_rm, reason = self.risk_manager.check_and_manage_position(current_price, current_timestamp)
            if closed_by_rm:
                closure_reason = reason

        realized_pnl_from_step = self.risk_manager.get_last_realized_pnl() if self.rm_closed_trade_this_step else 0.0
        
        if not self.rm_closed_trade_this_step:
            if action == 1 and self._position == 0:  # Buy/long
                self._open_position(current_price, current_timestamp, 'long')
            elif action == 2 and self._position == 0:  # Sell/short
                self._open_position(current_price, current_timestamp, 'short')
            elif action == 2 and self._position > 0:  # Sell to close long
                closure_reason = 'AGENT_ACTION_SELL_LONG'
                realized_pnl_from_step = self._close_position(current_price, current_timestamp, closure_reason)
            elif action == 1 and self._position < 0:  # Buy to cover short
                closure_reason = 'AGENT_ACTION_COVER_SHORT'
                realized_pnl_from_step = self._close_position(current_price, current_timestamp, closure_reason)
        
        reward = self._calculate_reward(realized_pnl_from_step, closure_reason, action)
        
        self.current_step += 1
        terminated = self.portfolio_value <= 0 or self.risk_manager.has_exceeded_max_drawdown() or self.current_step >= len(self.prices) - 1
        truncated = self.current_step >= self.max_steps - 1

        return self._get_obs(), reward, terminated, truncated, {}

    def _open_position(self, price, timestamp, position_type):
        position_size = self.risk_manager.calculate_position_size(price, self.portfolio_value)
        if position_size > 0:
            self._position = position_size if position_type == 'long' else -position_size
            self.entry_price = price
            self.risk_manager.open_position_calculations(price, position_type, timestamp, position_size)
            self.current_trade_open_details = {'entry_price': price, 'entry_timestamp': timestamp, 'trade_type': position_type, 'quantity': position_size}

    def _close_position(self, price, timestamp, reason):
        realized_pnl = self.calculate_realized_pnl(price)
        self.update_env_on_close(price, timestamp, realized_pnl, reason)
        self.risk_manager.reset()
        self._position = 0
        return realized_pnl

    def _get_obs(self):
        start = max(0, self.current_step - self.window_size)
        end = self.current_step
        obs_data = self.df.iloc[start:end].values
        
        if len(obs_data) < self.window_size:
            padding = np.zeros((self.window_size - len(obs_data), obs_data.shape[1]))
            obs_data = np.vstack((padding, obs_data))
            
        return obs_data.astype(np.float32)

    def _get_current_price(self):
        return self.prices[self.current_step]
        
    def calculate_realized_pnl(self, exit_price):
        if self.entry_price == 0: return 0
        pnl = (exit_price - self.entry_price) * self._position
        cost = (abs(self._position) * self.entry_price + abs(self._position) * exit_price) * self.risk_manager.config.transaction_cost
        return pnl - cost

    def _calculate_reward(self, realized_pnl: float, closure_reason: Optional[str] = None, action: int = 0) -> float:
        """
        Calculates a more sophisticated reward based on a combination of realized profit,
        risk management, and behavioral shaping.

        The reward structure is designed to:
        - Strongly reward realized profits.
        - Penalize realized losses to encourage risk aversion.
        - Penalize hitting a stop-loss.
        - Encourage taking profits by penalizing holding a winning position.
        - Gently penalize holding a losing position to discourage inaction.
        - Gently penalize inaction when no position is open to encourage exploration.
        """
        reward = 0.0

        # --- Novelty Reward (Intrinsic Motivation) ---
        obs_tuple = tuple(self._get_obs().flatten())
        if obs_tuple not in self.visited_states:
            reward += self.novelty_reward_scale
            self.visited_states.add(obs_tuple)

        # 1. Reward for Realized PnL (Profit and Loss)
        if realized_pnl != 0:
            if realized_pnl > 0:
                # Apply a bonus for profitable trades
                reward += realized_pnl * self.profit_incentive
            else:
                # Apply a penalty for losses, scaled by the reward_scale
                reward += realized_pnl * self.reward_scale

        # 2. Penalty for hitting stop-loss
        if closure_reason and 'stop_loss' in closure_reason:
            reward -= self.stop_loss_penalty

        # 3. Penalties for Holding (Unrealized PnL and Inaction)
        if realized_pnl == 0 and self._position != 0:
            # We are holding an open position
            current_price = self._get_current_price()
            unrealized_pnl = (current_price - self.entry_price) * self._position
            
            if unrealized_pnl > 0:
                # Holding a winning position: Penalize for not taking profits.
                # The penalty is a fraction of the unrealized PnL.
                reward -= (unrealized_pnl * 0.1) * self.holding_penalty
            else:
                # Holding a losing position: A small, constant penalty to discourage inaction.
                # Rewarding holding a losing trade is risky as it can lead to large drawdowns.
                # The stop-loss mechanism is the primary way to manage significant losses.
                reward -= self.holding_penalty

        # 4. Penalty for Inaction when Flat
        # A smaller penalty to encourage the agent to enter trades.
        if self._position == 0 and action == 0:
            reward -= self.holding_penalty * 0.1 # 10% of the holding penalty

        # Store returns for final metrics calculation
        if realized_pnl != 0:
            self.returns.append(realized_pnl)

        return reward

    def update_env_on_close(self, exit_price: float, exit_time, realized_pnl: float, closed_by: str) -> None:
        if self.current_trade_open_details:
            self.episode_trade_details.append({**self.current_trade_open_details, 'exit_price': exit_price, 'exit_timestamp': exit_time, 'pnl': realized_pnl, 'closure_reason': closed_by})
        self._balance += realized_pnl
        self.portfolio_value = self._balance
        self.rm_closed_trade_this_step = True
        self.current_trade_open_details = None

    def set_trade_exit_details(self, exit_price: float, exit_time, exit_reason: str) -> None:
        pass 

    def get_metrics(self) -> Dict[str, Any]:
        """Calculates and returns performance metrics for the episode."""
        if not self.episode_trade_details:
            return {}
        
        metrics_calculator = EnhancedTradingMetrics(risk_manager=self.risk_manager)
        metrics_calculator.process_episode_trades(self.episode_trade_details)
        performance_metrics = metrics_calculator.calculate_performance_metrics()
        
        return performance_metrics 

    def get_trade_log(self) -> List[Dict[str, Any]]:
        """Returns the log of trades for the current episode."""
        return self.episode_trade_details

    def get_holding_penalty(self) -> float:
        """Returns the current holding penalty."""
        return self.holding_penalty

    def set_holding_penalty(self, penalty: float) -> None:
        """Sets the holding penalty."""
        self.holding_penalty = penalty

    def reset_metrics(self) -> None:
        """Resets the metrics for a new evaluation episode."""
        self.episode_trade_details = []
        self.returns = [] 