from typing import List, Dict, Set, Callable
import pandas as pd
import numpy as np
from functools import partial
from technical_indicators import (
    calculate_momentum_indicators,
    calculate_price_channels,
    calculate_vwap,
    calculate_rsi,
    calculate_atr,
    calculate_sma,
    calculate_std_dev,
    calculate_price_div_sma
)
from logger_setup import logger

class TechnicalIndicatorManager:
    def __init__(self):
        self.active_indicators: Set[str] = set()
        self._available_indicators: Dict[str, Callable] = {
            'macd': calculate_momentum_indicators,
            'bollinger': calculate_price_channels,
            'vwap': calculate_vwap,
            'rsi': partial(calculate_rsi, window=14),
            'rsi_28': partial(calculate_rsi, window=28),
            'atr': partial(calculate_atr, window=14),
            'sma_50': partial(calculate_sma, window=50),
            'sma_200': partial(calculate_sma, window=200),
            'std_dev_14': partial(calculate_std_dev, window=14),
            'price_div_sma_50': partial(calculate_price_div_sma, window=50)
        }
        
    def add_indicator(self, indicator_name: str) -> bool:
        """Add a new technical indicator to the active set."""
        if indicator_name in self._available_indicators:
            self.active_indicators.add(indicator_name)
            logger.info(f"Added indicator: {indicator_name}")
            return True
        logger.warning(f"Attempted to add unavailable indicator: {indicator_name}")
        return False
        
    def remove_indicator(self, indicator_name: str) -> bool:
        """Remove a technical indicator from the active set."""
        if indicator_name in self.active_indicators:
            self.active_indicators.remove(indicator_name)
            logger.info(f"Removed indicator: {indicator_name}")
            return True
        logger.warning(f"Attempted to remove non-active or unavailable indicator: {indicator_name}")
        return False
        
    def calculate_indicators(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all active technical indicators and add them to the DataFrame."""
        if ohlcv_data.empty:
            logger.warning("TechnicalIndicatorManager.calculate_indicators received empty ohlcv_data. Returning empty DataFrame.")
            return pd.DataFrame()

        data_with_indicators = ohlcv_data.copy()
        
        for indicator_name in self.active_indicators:
            calc_func = self._available_indicators.get(indicator_name)
            if calc_func:
                try:
                    logger.debug(f"Calculating indicator: {indicator_name}")
                    indicator_output = calc_func(data_with_indicators)
                    
                    if isinstance(indicator_output, pd.DataFrame):
                        # For functions returning a DataFrame (like momentum)
                        data_with_indicators = data_with_indicators.join(indicator_output, how='left')
                    elif isinstance(indicator_output, pd.Series):
                        # For functions returning a Series (like rsi, vwap)
                        data_with_indicators[indicator_output.name] = indicator_output
                    else:
                        logger.warning(f"Indicator {indicator_name} returned an unexpected type: {type(indicator_output)}")
                except Exception as e:
                    logger.error(f"Failed to calculate indicator {indicator_name}: {e}", exc_info=True)
        
        # The join might create duplicate columns if names overlap; let's handle it, though refactored functions should prevent this.
        data_with_indicators = data_with_indicators.loc[:,~data_with_indicators.columns.duplicated()]

        # Final fillna
        data_with_indicators = data_with_indicators.ffill().bfill().fillna(0)
        
        return data_with_indicators
        
    def get_observation_space_size(self) -> int:
        """Get the total size of all active indicators' output."""
        sizes = {
            'macd': 3,  # macd, macd_signal, macd_hist from calculate_momentum_indicators
            'bollinger': 3,  # bb_upper, bb_middle, bb_lower from calculate_price_channels
            'vwap': 1,  # VWAP value
            'rsi': 1,  # RSI value
            'atr': 1,  # ATR value
            'sma_50': 1,  # SMA_50 value
            'sma_200': 1,  # SMA_200 value
            'std_dev_14': 1,  # STD_DEV_14 value
            'price_div_sma_50': 1  # Price_DIV_SMA_50 value
        }
        calculated_size = 0
        for ind_name in self.active_indicators:
            size = sizes.get(ind_name)
            if size is not None:
                calculated_size += size
            else:
                logger.warning(f"Size not defined for active indicator: {ind_name} in get_observation_space_size. It will be excluded from size calculation.")
        return calculated_size
        
    def evaluate_indicator_impact(self, 
                                indicator_name: str,
                                performance_history: List[float]) -> float:
        """
        Evaluate the impact of an indicator on trading performance.
        Returns a score between 0 and 1.
        """
        if not isinstance(performance_history, list) or len(performance_history) < 2:
            logger.warning("Performance history too short or invalid for evaluate_indicator_impact. Returning neutral score.")
            return 0.5
            
        try:
            perf_array = np.array(performance_history, dtype=np.float64)
            if np.isnan(perf_array).any() or np.isinf(perf_array).any():
                logger.warning("NaN or Inf in performance_history for evaluate_indicator_impact. Returning neutral score.")
                return 0.5
        except Exception as e:
            logger.warning(f"Error converting performance_history to numpy array: {e}. Returning neutral score.")
            return 0.5
        
        split_point = len(perf_array) // 2
        if split_point == 0 and len(perf_array) > 0:
             performance_before = 0.0
             performance_after = float(np.mean(perf_array))
        elif split_point == 0 and len(perf_array) == 0:
            return 0.5
        else:
            performance_before = float(np.mean(perf_array[:split_point]))
            performance_after = float(np.mean(perf_array[split_point:]))
        
        if abs(performance_before) < 1e-9:
            if abs(performance_after) < 1e-9:
                improvement = 0.0
            elif performance_after > 0:
                improvement = 1.0
            else:
                improvement = -1.0
        else:
            improvement = float((performance_after - performance_before) / abs(performance_before))
        
        score = float(np.clip(0.5 + (improvement / 2.0), 0.0, 1.0))
        logger.debug(f"Indicator impact for {indicator_name}: Before={performance_before:.4f}, After={performance_after:.4f}, Improvement={improvement:.4f}, Score={score:.4f}")
        return score
