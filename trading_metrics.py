# Standard library imports
from collections import deque
from datetime import datetime
import traceback
from typing import List, Optional, Dict, Any, Union, Deque, Callable, Type, cast

# Third-party imports
import numpy as np
import pandas as pd
from numpy.typing import NDArray

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

# Logger setup
from logger_setup import logger

class BaseTradingMetrics:
    """Base class for trading metrics"""
    def __init__(self) -> None:
        self.initialized = True

class TradingMetrics(BaseTradingMetrics):
    def __init__(self, risk_manager: Optional[Any] = None) -> None:
        """Initialize trading metrics with proper type handling."""
        super().__init__()
        

        # Initialize all attributes directly in __init__ for TradingMetrics
        # DO NOT call self.reset() from TradingMetrics.__init__ to avoid polymorphic calls
        # to subclass reset methods before subclass attributes are initialized.
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.total_profit: float = 0.0
        self.total_loss: float = 0.0
        self.peak_value: float = float(START_CAP)
        self.max_drawdown: float = 0.0
        self.max_consecutive_losses: int = 0
        self.consecutive_losses: int = 0
        self.trades: List[Dict[str, Any]] = []
        self.daily_returns: List[float] = []
        self.positions: List[float] = []
        self.value_history: List[float] = []
        self.drawdown_history: List[float] = []
        self.mae_history: List[float] = []
        self.mfe_history: List[float] = []
        self.last_trade_time: Optional[datetime] = None
        self.last_update_time: Optional[datetime] = None
        self.daily_values: Dict[str, float] = {}
        self.previous_day_value: float = float(START_CAP)
        self.daily_high: float = float(START_CAP)
        self.daily_low: float = float(START_CAP)

        self.risk_manager = risk_manager

        # Logging for RiskManager received (optional, can be kept)
        if risk_manager is not None:
            if not (hasattr(risk_manager, 'config') and hasattr(risk_manager.config, 'max_portfolio_risk')):
                logger.error(
                    f"RiskManager instance passed to TradingMetrics.__init__ is missing 'config.max_portfolio_risk'. "
                    f"Object type: {type(risk_manager)}, Config type: {type(getattr(risk_manager, 'config', None))}"
                )
            else:
                logger.info(f"TradingMetrics received RiskManager with config.max_portfolio_risk: {risk_manager.config.max_portfolio_risk}")
        else:
            logger.info("TradingMetrics initialized without a RiskManager instance.")

    def reset(self) -> None:
        """Reset all metrics to initial values for TradingMetrics."""
        # This method resets attributes defined in TradingMetrics.
        # It does NOT call super().reset() as BaseTradingMetrics has no reset method.
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.peak_value = float(START_CAP)
        self.max_drawdown = 0.0
        self.max_consecutive_losses = 0
        self.consecutive_losses = 0
        self.trades = []
        self.daily_returns = []
        self.positions = []
        self.value_history = []
        self.drawdown_history = []
        self.mae_history = []
        self.mfe_history = []
        self.last_trade_time = None
        self.last_update_time = None
        self.daily_values = {}
        self.previous_day_value = float(START_CAP)
        self.daily_high = float(START_CAP)
        self.daily_low = float(START_CAP)
        # logger.info("TradingMetrics attributes reset.") # Optional: logging

    def calculate_sharpe_ratio(self, daily_returns: NDArray[np.float64]) -> float:
        """Calculate Sharpe ratio with proper error handling."""
        try:
            returns_mean = np.mean(daily_returns)
            returns_std = np.std(daily_returns)
            if returns_std > 0:
                return float(returns_mean / returns_std * np.sqrt(252))
            return 0.0
        except Exception:
            return 0.0
            
    def calculate_sortino_ratio(self, daily_returns: NDArray[np.float64]) -> float:
        """Calculate Sortino ratio with proper error handling."""
        try:
            returns_mean = np.mean(daily_returns)
            negative_returns = daily_returns[daily_returns < 0]
            if len(negative_returns) > 0:
                downside_std = np.std(negative_returns)
                if downside_std > 0:
                    return float(returns_mean / downside_std * np.sqrt(252))
            return 0.0
        except Exception:
            return 0.0
            
    def safe_float_conversion(self, value: Any, default: float = 0.0) -> float:
        """Safely convert any value to float with proper type handling."""
        try:
            if value is None:
                return default
                
            # Handle numeric types directly
            if isinstance(value, (int, float, np.number)):
                return float(value)
                
            # Handle strings safely
            if isinstance(value, str):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
                    
            # Handle boolean values
            if isinstance(value, bool):
                return float(value)
                
            # Handle numpy arrays
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    return float(value.item())
                return default
                
            # Handle objects with __float__ protocol
            if hasattr(value, '__float__'):
                try:
                    result = float(value)
                    if np.isfinite(result):
                        return result
                    return default
                except (ValueError, TypeError, OverflowError):
                    return default
                    
            # Handle callable objects
            if callable(value):
                try:
                    result = value()
                    return self.safe_float_conversion(result, default)
                except:
                    return default
                    
            return default
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            return default
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculates and returns a dictionary of performance metrics based on the trades processed.
        This corrected version uses self.trades to derive returns for Sharpe/Sortino.
        """
        logger.info("[METRICS_DEBUG] Calculating performance metrics.")
        logger.info(f"[METRICS_DEBUG] State: total_trades={self.total_trades}, winning_trades={self.winning_trades}, losing_trades={self.losing_trades}")
        logger.info(f"[METRICS_DEBUG] State: total_profit={self.total_profit}, total_loss={self.total_loss}")
        logger.info(f"[METRICS_DEBUG] Number of trades in self.trades list: {len(self.trades)}")

        metrics: Dict[str, Any] = {
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': self.max_drawdown,
            'avg_trade': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'risk_reward_ratio': 0.0,
            'avg_trade_duration_steps': 0.0
        }

        if not self.trades:
            logger.warning("[METRICS_DEBUG] No trades recorded, returning zeroed metrics.")
            return metrics

        # --- Calculate returns from the detailed trade list for Sharpe and Sortino ---
        pnl_returns = [trade.get('pnl_percentage', 0.0) for trade in self.trades if 'pnl_percentage' in trade]
        logger.info(f"[METRICS_DEBUG] Found {len(pnl_returns)} trades with 'pnl_percentage' for Sharpe/Sortino calculation.")
        
        if len(pnl_returns) > 1:
            pnl_returns_arr = np.array(pnl_returns, dtype=np.float64)
            
            # Sharpe Ratio
            std_dev = np.std(pnl_returns_arr)
            if std_dev > 0:
                sharpe = np.mean(pnl_returns_arr) / std_dev * np.sqrt(252)
                metrics['sharpe_ratio'] = float(sharpe) if np.isfinite(sharpe) else 0.0

            # Sortino Ratio
            negative_returns = pnl_returns_arr[pnl_returns_arr < 0]
            if len(negative_returns) > 1:
                downside_std = np.std(negative_returns)
                if downside_std > 0:
                    sortino = np.mean(pnl_returns_arr) / downside_std * np.sqrt(252)
                    metrics['sortino_ratio'] = float(sortino) if np.isfinite(sortino) else 0.0

        # --- Aggregate calculations ---
        if self.total_trades > 0:
            metrics['win_rate'] = self.winning_trades / self.total_trades
            if self.total_loss != 0:
                metrics['profit_factor'] = self.total_profit / abs(self.total_loss)
            elif self.total_profit > 0:
                metrics['profit_factor'] = 999.0  # High value to represent no losses
            else:
                metrics['profit_factor'] = 0.0
            metrics['avg_trade'] = (self.total_profit + self.total_loss) / self.total_trades
        
        if self.winning_trades > 0:
            metrics['avg_win'] = self.total_profit / self.winning_trades
        
        if self.losing_trades > 0:
            metrics['avg_loss'] = self.total_loss / self.losing_trades
            if metrics['avg_loss'] != 0:
                metrics['risk_reward_ratio'] = metrics['avg_win'] / abs(metrics['avg_loss'])
        
        # Calculate avg trade duration
        total_duration = sum(trade.get('duration_steps', 0) for trade in self.trades)
        if self.total_trades > 0:
            metrics['avg_trade_duration_steps'] = total_duration / self.total_trades

        logger.info(f"[METRICS_DEBUG] Finished calculating metrics: {metrics}")
        return metrics
        
    def record_trade(self,
                    trade_result: float,
                    current_value: float,
                    position_size: float,
                    trade_type: str,
                    entry_time: Optional[datetime],
                    exit_time: Optional[datetime],
                    pnl: float,
                    transaction_cost: float,
                    entry_price: Optional[float] = None,
                    exit_price: Optional[float] = None,
                    reward: Optional[float] = None,
                    portfolio_value: Optional[Union[float, Callable[[], float]]] = None,
                    timestamp: Optional[datetime] = None,
                    info: Optional[Dict[str, Any]] = None) -> None:
        """
        Records a single trade and updates all relevant metrics.
        Should be called when a position is opened, closed, or modified significantly.
        """
        logger.info(f"[METRICS_DEBUG] TradingMetrics.record_trade called. Current total_trades before this call: {self.total_trades}. Trade Type: {trade_type}, PnL: {pnl}")

        try:
            # Validate and convert inputs
            pnl = float(pnl)
            current_value = float(current_value)
            position_size = float(position_size)
            transaction_cost = float(transaction_cost)
            
            if portfolio_value is not None:
                if callable(portfolio_value):
                    try:
                        portfolio_value = float(portfolio_value())
                    except (TypeError, ValueError):
                        portfolio_value = current_value
                else:
                    try:
                        portfolio_value = float(portfolio_value)
                    except (TypeError, ValueError):
                        portfolio_value = current_value
            else:
                portfolio_value = current_value
                
            # Update core metrics
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1
                self.total_profit += pnl
                self.consecutive_losses = 0
            elif pnl < 0:
                self.losing_trades += 1
                self.total_loss += pnl
                self.consecutive_losses += 1
                self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
                
            # Update advanced metrics
            self.peak_value = max(self.peak_value, portfolio_value)
            current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Record trade details
            entry_time_iso = None
            if entry_time:
                if isinstance(entry_time, np.datetime64):
                    entry_time_iso = pd.Timestamp(entry_time).isoformat()
                elif hasattr(entry_time, 'isoformat'):
                    entry_time_iso = entry_time.isoformat()
                elif isinstance(entry_time, datetime):
                    entry_time_iso = entry_time.isoformat()
                else:
                    entry_time_iso = str(entry_time)

            exit_time_iso = None
            if exit_time:
                if isinstance(exit_time, np.datetime64):
                    exit_time_iso = pd.Timestamp(exit_time).isoformat()
                elif hasattr(exit_time, 'isoformat'):
                    exit_time_iso = exit_time.isoformat()
                elif isinstance(exit_time, datetime):
                    exit_time_iso = exit_time.isoformat()
                else:
                    exit_time_iso = str(exit_time)

            pnl_percentage = 0.0
            capital_at_risk = entry_price * abs(position_size) if entry_price is not None else 0
            if capital_at_risk > 0:
                pnl_percentage = (pnl / capital_at_risk) * 100

            trade_info = {
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position_size': position_size,
                'trade_type': trade_type,
                'pnl': pnl,
                'pnl_percentage': pnl_percentage,
                'transaction_cost': transaction_cost,
                'entry_time': entry_time_iso,
                'exit_time': exit_time_iso,
                'reward': reward,
                'portfolio_value_at_exit': portfolio_value,
                'drawdown': current_drawdown,
                'info': info or {}
            }
            self.trades.append(trade_info)
            
            # Update history lists
            self.value_history.append(portfolio_value)
            self.positions.append(position_size)
            self.drawdown_history.append(current_drawdown)
            
            # Calculate and store performance metrics
            self.performance_metrics = self.calculate_performance_metrics()
            
            # Update time tracking
            self.last_trade_time = exit_time or datetime.now()
            self.last_update_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating trade metrics: {str(e)}")
            traceback.print_exc()
            
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate all trading metrics with proper type handling."""
        try:
            metrics: Dict[str, float] = {}
            
            # Basic metrics
            if self.total_trades > 0:
                metrics['win_rate'] = self.win_rate
                metrics['average_win'] = self.average_win
                metrics['average_loss'] = self.average_loss
            
            # Returns and ratios
            if len(self.daily_returns) > 0:
                returns_array = np.array(self.daily_returns, dtype=np.float64)
                returns_std = float(np.std(returns_array))
                if returns_std > 0:
                    metrics['sharpe_ratio'] = float(np.mean(returns_array) / returns_std * np.sqrt(252))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}

    @property
    def win_rate(self) -> float:
        """Calculate win rate with proper error handling."""
        if self.total_trades > 0:
            return float(self.winning_trades) / float(self.total_trades)
        return 0.5  # Default to neutral win rate
        
    @property
    def average_win(self) -> float:
        """Calculate average winning trade size."""
        if self.winning_trades > 0:
            return float(self.total_profit) / float(self.winning_trades)
        return 0.02  # Default 2% win
        
    @property
    def average_loss(self) -> float:
        """Calculate average losing trade size."""
        if self.losing_trades > 0:
            return float(self.total_loss) / float(self.losing_trades)
        return 0.01  # Default 1% loss

class EnhancedTradingMetrics(TradingMetrics):
    """Enhanced trading metrics with more detailed tracking."""
    def __init__(self, risk_manager: Optional[Any] = None) -> None:
        # Initialize EnhancedTradingMetrics-specific attributes FIRST
        self.rewards: Deque[float] = deque(maxlen=1000) 
        self.portfolio_values: Deque[float] = deque(maxlen=1000)
        self.total_steps_in_trades: int = 0
        self.closed_trades_count_for_duration: int = 0
        
        # Now call parent __init__. This will initialize parent attributes.
        # TradingMetrics.__init__ should NOT call self.reset().
        super().__init__(risk_manager) 
        
        # After parent and child attributes are declared/initialized (parent via super init, child here),
        # call self.reset() to ensure all states are set to their reset values.
        # This reset will call TradingMetrics.reset(self) and then reset child-specific attributes.
        self.reset() 

    def reset(self) -> None:
        """Reset all metrics to initial states for EnhancedTradingMetrics."""
        # Call the parent's reset method directly and explicitly.
        TradingMetrics.reset(self)
        # Now reset attributes specific to EnhancedTradingMetrics
        self.rewards.clear() 
        self.portfolio_values.clear()
        self.total_steps_in_trades = 0
        self.closed_trades_count_for_duration = 0
        logger.info("EnhancedTradingMetrics have been reset.")

    def process_episode_trades(self, episode_trade_details: List[Dict[str, Any]]) -> None:
        """
        Processes a list of completed trade dictionaries from an evaluation episode,
        calculating the PnL for each trade and then calling the record_trade method
        to ensure all metrics are correctly updated.
        """
        if not episode_trade_details:
            return

        sorted_trades = sorted(episode_trade_details, key=lambda x: x.get('exit_step', 0))

        for trade in sorted_trades:
            entry_price = trade.get('entry_price', 0.0)
            exit_price = trade.get('exit_price', 0.0)
            quantity = trade.get('quantity', 0.0)
            trade_type = trade.get('trade_type', 'unknown')

            # Robustly calculate PnL
            if trade_type == 'long':
                pnl = (exit_price - entry_price) * quantity
            elif trade_type == 'short':
                pnl = (entry_price - exit_price) * quantity
            else:
                pnl = 0.0

            # Use the record_trade method with the correctly calculated PnL
            self.record_trade(
                trade_result=pnl,
                current_value=trade.get('portfolio_value_at_exit', self.current_value + pnl),
                position_size=quantity,
                trade_type=trade_type,
                entry_time=trade.get('entry_timestamp'),
                exit_time=trade.get('exit_timestamp'),
                pnl=pnl,
                transaction_cost=trade.get('transaction_cost', 0.0),
                entry_price=entry_price,
                exit_price=exit_price,
                reward=trade.get('reward'),
                portfolio_value=trade.get('portfolio_value_at_exit'),
                timestamp=trade.get('exit_timestamp'),
                info=trade.get('info', {})
            )

    def update(self, reward: float, portfolio_value: float, position: int, 
               timestamp: Optional[datetime] = None) -> None:
        """Update trading metrics."""
        try:
            # Convert values to proper types
            reward = float(reward)
            portfolio_value = float(portfolio_value)
            position = int(position)
            
            # Update peak value and drawdown (This is fine and general)
            self.peak_value = max(self.peak_value, portfolio_value)
            current_drawdown = 1.0 - (portfolio_value / self.peak_value) if self.peak_value > 0 else 0.0
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Store history using attributes from parent TradingMetrics or self
            if hasattr(self, 'value_history') and isinstance(self.value_history, list):
                self.value_history.append(portfolio_value)
            if hasattr(self, 'drawdown_history') and isinstance(self.drawdown_history, list):
                self.drawdown_history.append(current_drawdown)
            if hasattr(self, 'portfolio_values') and hasattr(self.portfolio_values, 'append'): # For deque
                self.portfolio_values.append(portfolio_value)
            if hasattr(self, 'rewards') and hasattr(self.rewards, 'append'): # For deque
                self.rewards.append(reward)
            
            # Block that attempted to use self.trade_history has been removed as per previous instructions.
            # If step-wise reward/event logging is needed, a new, properly initialized attribute should be used.
                
            # Update risk manager if available
            if self.risk_manager is not None:
                if hasattr(self.risk_manager, 'update_peak'):
                    self.risk_manager.update_peak(portfolio_value)
                
                if hasattr(self.risk_manager, 'update_market_conditions'):
                    # Example: self.risk_manager.update_market_conditions(price=current_price_at_timestamp, volatility=current_vol_at_timestamp)
                    pass # Actual call would depend on what data RM.update_market_conditions needs
                
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
            traceback.print_exc()
    
    def update_portfolio_value(self, value: float) -> None: # Added type hint for value
        """Update the current portfolio value history"""
        # self.portfolio_value = value # REMOVED: Attribute not consistently defined/used for current value storage here
        if hasattr(self, 'value_history') and isinstance(self.value_history, list):
            self.value_history.append(value)
        if hasattr(self, 'portfolio_values') and hasattr(self.portfolio_values, 'append'): # For deque
            self.portfolio_values.append(value)
    
    @property
    def current_value(self) -> float: # Added return type hint
        """Get current portfolio value from history."""
        if self.portfolio_values:
            return self.portfolio_values[-1]
        return float(START_CAP) # Default if no history

    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculates and returns a dictionary of performance metrics based on the trades processed.
        This corrected version uses self.trades to derive returns for Sharpe/Sortino.
        """
        # This method in the subclass can override or extend the base class method.
        # For now, it calls the superclass method, but could add more metrics specific to this class.
        # super().calculate_performance_metrics() can be called if this method extends it.
        base_metrics = super().calculate_performance_metrics()

        # Add or override metrics specific to EnhancedTradingMetrics if any
        # For example, could add metrics related to risk manager interactions if tracked
        
        return base_metrics