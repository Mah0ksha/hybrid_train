from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timedelta
from typing import Tuple, Optional, Any, Union
import numpy as np
import pandas as pd

# Import Constants
from config_vals import (
    CURRENT_UTC_TIME, CURRENT_USER, TIME_FILTERS, UNIVERSE_FILE, RAW_DIR, 
    START_CAP, TRANSACTION_COST, MAX_DRAWDOWN, RISK_PER_TRADE, STOP_LOSS, 
    MAX_POSITION_SIZE, TRAILING_STOP_PERCENT, SCALE_IN_LEVELS, SCALE_OUT_LEVELS, 
    MAX_POSITION_DURATION, MIN_LIQUIDITY_THRESHOLD, DYNAMIC_POSITION_SCALING, 
    MAX_LEVERAGE, MIN_POSITION_SIZE, POSITION_STEP_SIZE, REWARD_SCALE, 
    MAX_POSITION_DURATION_HOURS, MIN_TRADE_INTERVAL_MINUTES, 
    OPTIMAL_POSITION_SIZE_MIN, OPTIMAL_POSITION_SIZE_MAX, BASE_TIMESTEPS, 
    MAX_TIMESTEPS, PERFORMANCE_THRESHOLD, CHUNK_SIZE, MARKET_HOURS, 
    INPUT_DIM, SEQUENCE_LENGTH, WINDOW_SIZE, PREDICTION_WINDOW, EVAL_DAYS, 
    VOLATILITY_ADJUSTMENT, BATCH_SIZE, LEARNING_RATE, GAMMA, TAU, 
    ENTROPY_COEF, VF_COEF, MAX_GRAD_NORM, N_STEPS, N_EPOCHS, N_ENVS, 
    RL_TIMESTEPS, EVAL_STEPS, WARMUP_STEPS, PROGRESS_INTERVAL
)

# Import logger first to ensure it's available for other imports
from logger_setup import logger, setup_logger

# Import RiskConfig
from risk_config import RiskConfig

# Re-initialize logger with custom settings
logger = setup_logger()

# Define the interface for TradingEnv to implement
class RiskManagerEnvInterface(ABC):
    @abstractmethod
    def update_env_on_close(self, exit_price: float, exit_time: datetime, realized_pnl: float, closed_by: str) -> None:
        """Callback to update the TradingEnv\'s state when a position is closed by RiskManager."""
        pass

    @abstractmethod
    def set_trade_exit_details(self, exit_price: float, exit_time: datetime, exit_reason: str) -> None:
        """Callback to set specific exit details in TradingEnv\'s metrics or state for the closed trade."""
        pass

class RiskManager:
    """
    Refactored risk manager to work with RiskConfig and an environment interface.
    
    Args:
        config: RiskConfig object
        env_interface: Optional RiskManagerEnvInterface object
        initial_capital: float, optional initial capital for the risk manager
    """
    def __init__(
        self,
        config: RiskConfig,
        env_interface: Optional[RiskManagerEnvInterface] = None,
        initial_capital: float = 100000.0 # Can be part of config or passed if needed
    ):
        self.config = config
        self.env_interface = env_interface
        
        # Portfolio tracking (simplified, actual balance/value managed by env)
        self.initial_capital = float(initial_capital)
        self.portfolio_value = float(initial_capital) # May not be needed if env handles it
        self.peak_value = float(initial_capital) # For drawdown calculations if RM handles them
        self.current_drawdown = 0.0

        # Core position state
        self.entry_price: Optional[float] = None
        self.entry_time: Optional[datetime] = None
        self.stop_loss_price: Optional[float] = None
        self.take_profit_price: Optional[float] = None
        self.current_position_size_abs: float = 0.0  # Absolute quantity of shares/contracts
        self.current_position_type: Optional[str] = None  # 'long' or 'short'
        self.last_realized_pnl = 0.0
        
        # Example: other attributes from old RM if still relevant and not in RiskConfig
        # self.transaction_cost = self.config.transaction_cost_pct # If RiskConfig has transaction_cost_pct
        # self.trailing_stop_pct = self.config.trailing_stop_pct # If RiskConfig has this

        logger.info(f"RiskManager initialized with config: {self.config.to_dict()}")
        self.reset() # Initialize all position-specific states

    def reset(self) -> None:
        """Reset all position-specific states of the RiskManager."""
        self.entry_price = None
        self.entry_time = None
        self.stop_loss_price = None
        self.take_profit_price = None
        self.current_position_size_abs = 0.0
        self.current_position_type = None
        self.last_realized_pnl = 0.0
        
        # Reset other relevant states if any
        # self.highest_price_since_entry = None
        # self.lowest_price_since_entry = None
        logger.debug("RiskManager state has been reset.")

    def open_position_calculations(
        self,
        entry_price: float,
        position_type: str,
        entry_time: datetime,
        position_size_abs: float # Actual quantity determined by agent/env
    ) -> None:
        """
        Calculates and sets SL/TP prices when a new position is opened.
        Called by TradingEnv after it decides to open a position.
        """
        if position_type not in ['long', 'short']:
            logger.error(f"Invalid position_type: {position_type}")
            return
        if entry_price <= 0:
            logger.error(f"Invalid entry_price: {entry_price}")
            return
        if position_size_abs <= 0:
            logger.error(f"Invalid position_size_abs: {position_size_abs}")
            return

        self.entry_price = entry_price
        self.entry_time = entry_time
        self.current_position_type = position_type
        self.current_position_size_abs = position_size_abs

        if position_type == 'long':
            if self.config.base_stop_loss > 0:
                self.stop_loss_price = entry_price * (1 - self.config.base_stop_loss)
            else:
                self.stop_loss_price = None # No stop loss
            if self.config.base_take_profit > 0:
                self.take_profit_price = entry_price * (1 + self.config.base_take_profit)
            else:
                self.take_profit_price = None # No take profit
        elif position_type == 'short':
            if self.config.base_stop_loss > 0:
                self.stop_loss_price = entry_price * (1 + self.config.base_stop_loss)
            else:
                self.stop_loss_price = None # No stop loss
            if self.config.base_take_profit > 0:
                self.take_profit_price = entry_price * (1 - self.config.base_take_profit)
            else:
                self.take_profit_price = None # No take profit
        
        logger.info(
            f"RM: Position opened. Type: {self.current_position_type}, Entry: {self.entry_price:.4f}, "
            f"Size: {self.current_position_size_abs:.4f}, SL: {(f'{self.stop_loss_price:.4f}' if self.stop_loss_price is not None else 'N/A')}, "
            f"TP: {(f'{self.take_profit_price:.4f}' if self.take_profit_price is not None else 'N/A')}"
        )

    def open_long(self, entry_price: float, position_size: float):
        self.open_position_calculations(entry_price, 'long', datetime.now(), position_size)

    def open_short(self, entry_price: float, position_size: float):
        self.open_position_calculations(entry_price, 'short', datetime.now(), position_size)

    def close_position(self, exit_price: float) -> float:
        """Public method to close a position, callable from the environment."""
        return self._close_position(exit_price, datetime.now(), "AGENT_ACTION")

    def _close_position(self, exit_price: float, exit_time: datetime, reason: str) -> float:
        """
        Internal method to handle position closure logic.
        Calculates PnL and notifies the environment.
        Returns realized PnL for this specific closure.
        """
        if self.entry_price is None or self.current_position_type is None or self.current_position_size_abs == 0:
            logger.warning("RM: _close_position called but no active position or entry price.")
            return 0.0

        realized_pnl = 0.0
        transaction_cost_per_unit = self.config.transaction_cost if hasattr(self.config, 'transaction_cost') else 0.001 # Get from config
        
        # Calculate PnL based on position type
        if self.current_position_type == 'long':
            realized_pnl = (exit_price - self.entry_price) * self.current_position_size_abs
        elif self.current_position_type == 'short':
            realized_pnl = (self.entry_price - exit_price) * self.current_position_size_abs
        
        # Deduct transaction costs for both opening and closing trades
        # Assuming position_size_abs is the quantity traded.
        total_transaction_cost = (self.entry_price * self.current_position_size_abs * transaction_cost_per_unit) + \
                                 (exit_price * self.current_position_size_abs * transaction_cost_per_unit)
        realized_pnl -= total_transaction_cost

        self.last_realized_pnl = realized_pnl

        logger.info(
            f"RM: Position closed. Type: {self.current_position_type}, Entry: {self.entry_price:.4f}, "
            f"Exit: {exit_price:.4f}, Size: {self.current_position_size_abs:.4f}, Reason: {reason}, PnL: {realized_pnl:.4f}"
        )

        # Notify environment through the interface
        if self.env_interface:
            self.env_interface.update_env_on_close(exit_price, exit_time, realized_pnl, reason)
            self.env_interface.set_trade_exit_details(exit_price, exit_time, reason)
        else:
            logger.warning("RM: env_interface not set. Cannot notify environment of position closure.")

        # Reset internal state for the closed position
        self.reset() # This clears all position details

        return realized_pnl

    def get_stop_loss(self) -> Optional[float]:
        """Returns the current stop loss price."""
        return self.stop_loss_price

    def get_take_profit(self) -> Optional[float]:
        """Returns the current take profit price."""
        return self.take_profit_price

    def get_last_realized_pnl(self) -> float:
        return self.last_realized_pnl

    def check_and_manage_position(self, current_price: float, current_time: datetime) -> Tuple[bool, Optional[str]]:
        return self.check_stop_loss_take_profit(current_price, current_time)

    def check_stop_loss_take_profit(
        self,
        current_price: float,
        current_time: datetime # Added current_time
    ) -> Tuple[bool, Optional[str]]:
        """
        Checks if stop-loss or take-profit conditions are met.
        Returns: (position_was_closed, closure_reason)
        """
        if self.current_position_type is None or self.entry_price is None:
            return False, None # No active position

        # --- BEGIN DIAGNOSTIC LOGGING FOR SL/TP ---
        logger.debug(
            f"[RM_SLTP_CHECK] PosType: {self.current_position_type}, CurPrice: {current_price:.4f}, "
            f"Entry: {(f'{self.entry_price:.4f}' if self.entry_price is not None else 'N/A')}, " # Entry price can also be None before a position
            f"ConfSL%: {self.config.base_stop_loss:.4f}, CalcSL: {(f'{self.stop_loss_price:.4f}' if self.stop_loss_price is not None else 'N/A')}, "
            f"ConfTP%: {self.config.base_take_profit:.4f}, CalcTP: {(f'{self.take_profit_price:.4f}' if self.take_profit_price is not None else 'N/A')}"
        )
        # --- END DIAGNOSTIC LOGGING FOR SL/TP ---

        if self.current_position_type == 'long':
            # Check Take Profit for Long
            if self.take_profit_price is not None and current_price >= self.take_profit_price:
                logger.info(f"[RM_SLTP_TRIGGER] Long TP Triggered! Price {current_price:.4f} >= TP Price {self.take_profit_price:.4f}. Entry: {self.entry_price:.4f}")
                self._close_position(self.take_profit_price, current_time, "take_profit_long") # Close at TP price
                return True, "take_profit_long"

            # Check Stop Loss for Long
            if self.stop_loss_price is not None and current_price <= self.stop_loss_price:
                logger.info(f"[RM_SLTP_TRIGGER] Long SL Triggered! Price {current_price:.4f} <= SL Price {self.stop_loss_price:.4f}. Entry: {self.entry_price:.4f}")
                self._close_position(self.stop_loss_price, current_time, "stop_loss_long") # Close at SL price
                return True, "stop_loss_long"

        elif self.current_position_type == 'short':
            # Check Take Profit for Short
            if self.take_profit_price is not None and current_price <= self.take_profit_price:
                logger.info(f"[RM_SLTP_TRIGGER] Short TP Triggered! Price {current_price:.4f} <= TP Price {self.take_profit_price:.4f}. Entry: {self.entry_price:.4f}")
                self._close_position(self.take_profit_price, current_time, "take_profit_short") # Close at TP price
                return True, "take_profit_short"

            # Check Stop Loss for Short
            if self.stop_loss_price is not None and current_price >= self.stop_loss_price:
                logger.info(f"[RM_SLTP_TRIGGER] Short SL Triggered! Price {current_price:.4f} >= SL Price {self.stop_loss_price:.4f}. Entry: {self.entry_price:.4f}")
                self._close_position(self.stop_loss_price, current_time, "stop_loss_short") # Close at SL price
                return True, "stop_loss_short"

        return False, None # No SL/TP was hit

    # --- Methods from old RiskManager to be reviewed/removed/adapted ---
    # The following methods are placeholders or illustrative of old functionality.
    # They need to be critically reviewed. Many might be redundant if TradingEnv
    # and the new RiskManager structure handle their responsibilities.
    def calculate_position_size(self, price: float, portfolio_value: float) -> float:
        """
        Calculate position size based on risk parameters.
        """
        if price <= 0 or portfolio_value <= 0:
            return 0.0
        
        # Using risk_per_trade from config
        risk_amount = portfolio_value * self.config.risk_per_trade
        
        # Calculate stop loss to determine risk per share
        stop_loss_pct = self.config.base_stop_loss
        risk_per_share = price * stop_loss_pct
        
        if risk_per_share == 0:
            return 0.0
            
        position_size = risk_amount / risk_per_share
        
        # Max position size constraint
        max_position_value = portfolio_value * self.config.max_position_size
        max_allowed_size = max_position_value / price if price > 0 else 0.0
        
        return min(position_size, max_allowed_size)

    # The following methods are mostly related to internal state or calculations
    # that might be handled differently now or are illustrative.

    def _calculate_time_diff_minutes(self, time1: Union[datetime, int, float, str, None], time2: Union[datetime, int, float, str, None]) -> float:
        """Calculate time difference in minutes, handling different time types."""
        try:
            if time1 is None or time2 is None: return 0.0
            if isinstance(time1, str): time1 = datetime.fromisoformat(time1.replace('Z', '+00:00'))
            elif isinstance(time1, (int, float)): time1 = datetime.fromtimestamp(time1)
            if isinstance(time2, str): time2 = datetime.fromisoformat(time2.replace('Z', '+00:00'))
            elif isinstance(time2, (int, float)): time2 = datetime.fromtimestamp(time2)
            if not isinstance(time1, datetime) or not isinstance(time2, datetime):
                logger.warning(f"Could not convert times to datetime objects: {type(time1)}, {type(time2)}")
                return 0.0
            diff: timedelta = time1 - time2
            return diff.total_seconds() / 60.0
        except Exception as e:
            logger.warning(f"Error calculating time difference: {e}")
            return 0.0

    def update_peak(self, current_value: float) -> None: # If RM tracks its own drawdown
        """Update peak portfolio value and calculate current drawdown."""
        self.peak_value = max(self.peak_value, current_value)
        # self.portfolio_value = float(current_value) # Env should be source of truth for this
        self.current_drawdown = (self.peak_value - current_value) / self.peak_value if self.peak_value > 0 else 0.0
        logger.debug(f"RM: Peak updated. Peak: {self.peak_value}, Current: {current_value}, Drawdown: {self.current_drawdown}")

    # Other methods like adjust_for_volatility, update_consecutive_losses, get_metrics, etc.
    # from the old RiskManager would need to be reviewed.
    # Many metrics might now be primarily calculated by TradingMetrics/EnhancedTradingMetrics in the env.
    # RiskManager could still have its own internal metrics if useful for adaptive logic.

    def update_market_conditions(self, price: Optional[float] = None, volume: Optional[float] = None, volatility: Optional[float] = None) -> None:
        """
        Placeholder for updating RM based on market conditions, if needed for adaptive logic.
        Could use volatility to adjust SL/TP dynamically if RiskConfig supports it.
        """
        if volatility is not None and hasattr(self.config, 'volatility_scaling_factor'):
            # Example: could adjust effective SL/TP percentages based on volatility
            # This logic would need to be defined in RiskConfig or here.
            pass
        logger.debug(f"RM: Market conditions updated. Price: {price}, Vol: {volatility}")

    # Ensure all essential methods for the new API (init, reset, open_pos_calc, check_sltp, _close_pos) are robust.
    # Remaining methods from the old version should be integrated or removed.

    def has_exceeded_max_drawdown(self) -> bool:
        """Check if the maximum drawdown has been exceeded."""
        if self.config.max_portfolio_risk is not None:
            return self.current_drawdown > self.config.max_portfolio_risk
        return False