# Standard library imports
from datetime import datetime, timedelta
import traceback
from typing import Optional, Dict, Any, List, Union, cast, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from pandas.api.types import is_numeric_dtype
import numpy.typing as npt
import talib as ta

# Logger setup
from logger_setup import setup_logger
# Initialize logger
logger = setup_logger()

# Default window sizes for indicators
RSI_WINDOW = 14
ATR_WINDOW = 14
VOLATILITY_WINDOW = 20
MACD_FAST_WINDOW = 12
MACD_SLOW_WINDOW = 26
MACD_SIGNAL_WINDOW = 9
BB_WINDOW = 20
BB_STD_DEV = 2
ADX_WINDOW = 14
PRICE_CHANNEL_WINDOW = 20
TRADING_RANGE_WINDOW = 14

def safe_fillna(series: pd.Series, fill_forward: bool = True, fill_value: float | None = 0.0) -> pd.Series:
    """Safely handle fillna operations with proper type casting."""
    result = series
    if fill_forward:
        # Implement forward fill manually to avoid type issues
        mask = result.isna()
        result = result.copy()
        last_valid = None
        for i in range(len(result)):
            if mask[i]:
                if last_valid is not None:
                    result.iloc[i] = last_valid
            else:
                last_valid = result.iloc[i]
    if fill_value is not None:
        result = result.fillna(fill_value)
    return result

def get_market_calendar() -> pd.DatetimeIndex:
    """Get market calendar excluding weekends."""
    try:
        # Using pandas date_range and manually filtering for common holidays
        all_dates = pd.date_range(start='2020-01-01', end='2026-12-31', freq='D')
        # Filter out weekends
        business_days = pd.bdate_range(start='2020-01-01', end='2026-12-31')
        holiday_dates = sorted(set(all_dates) - set(business_days))
        return pd.DatetimeIndex(holiday_dates)
    except Exception as e:
        logger.error(f"Error getting market calendar: {str(e)}")
        return pd.DatetimeIndex([], dtype='datetime64[ns]')

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate Volume Weighted Average Price (VWAP)."""
    try:
        vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        vwap = vwap.ffill()
        vwap = vwap.bfill()
        return vwap.rename('vwap')
    except KeyError as e:
        logger.error(f"Error calculating VWAP: Missing column {e}. Available: {df.columns.tolist()}")
        return pd.Series(np.nan, index=df.index, name='vwap')
    except Exception as e:
        logger.error(f"Unexpected error calculating VWAP: {e}", exc_info=True)
        return pd.Series(np.nan, index=df.index, name='vwap')

def calculate_momentum_indicators(df: pd.DataFrame, window_size: int = 14) -> pd.DataFrame:
    if df.empty:
        logger.warning("Input DataFrame for momentum indicators is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    df_out = pd.DataFrame(index=df.index)

    # Ensure required 'close' column exists
    if 'close' not in df.columns:
        logger.error("Momentum Indicators: 'close' column missing. Cannot calculate indicators.")
        # Add NaN columns for expected outputs so downstream doesn't KeyError
        for col in ['rsi', 'macd', 'macd_signal', 'macd_hist']:
            df_out[col] = np.nan
        return df_out

    try:
        # Ensure 'close' is float for TA-Lib
        close_prices_series = df['close'].astype(float)
        close_prices_numpy = close_prices_series.to_numpy() # Convert to NumPy array
    except ValueError as ve:
        logger.error(f"Momentum Indicators: Could not convert 'close' to float: {ve}. Filling indicators with NaN.")
        for col in ['rsi', 'macd', 'macd_signal', 'macd_hist']:
            df_out[col] = np.nan
        return df_out
    except Exception as e_conv: # Catch any other conversion error
        logger.error(f"Momentum Indicators: Error converting 'close' to float: {e_conv}. Filling indicators with NaN.")
        for col in ['rsi', 'macd', 'macd_signal', 'macd_hist']:
            df_out[col] = np.nan
        return df_out


    # RSI
    try:
        df_out['rsi'] = ta.RSI(close_prices_numpy, timeperiod=window_size)
    except Exception as e:
        logger.warning(f"Could not calculate RSI: {e}. Filling 'rsi' with NaN.")
        df_out['rsi'] = np.nan # Ensures column exists

    # MACD
    try:
        macd, macdsignal, macdhist = ta.MACD(close_prices_numpy, fastperiod=12, slowperiod=26, signalperiod=9)
        df_out['macd'] = macd
        df_out['macd_signal'] = macdsignal
        df_out['macd_hist'] = macdhist
    except Exception as e:
        logger.warning(f"Could not calculate MACD: {e}. Filling MACD related columns with NaN.")
        df_out['macd'] = np.nan
        df_out['macd_signal'] = np.nan
        df_out['macd_hist'] = np.nan
        
    return df_out

def calculate_price_channels(df: pd.DataFrame, 
                             bb_window: int = BB_WINDOW, 
                             bb_std: int = BB_STD_DEV,
                             pc_window: int = PRICE_CHANNEL_WINDOW) -> pd.DataFrame:
    """Calculate Bollinger Bands and Price Channels (Donchian variant)."""
    df_out = pd.DataFrame(index=df.index)
    try:
        # Bollinger Bands
        df_out['bb_middle'] = df['close'].rolling(window=bb_window).mean()
        df_out['bb_std'] = df['close'].rolling(window=bb_window).std()
        df_out['bb_upper'] = df_out['bb_middle'] + (df_out['bb_std'] * bb_std)
        df_out['bb_lower'] = df_out['bb_middle'] - (df_out['bb_std'] * bb_std)
        # Fill initial NaNs for Bollinger Bands
        for col in ['bb_middle', 'bb_std', 'bb_upper', 'bb_lower']:
            df_out[col] = df_out[col].ffill()
            df_out[col] = df_out[col].bfill()
            if col in ['bb_middle', 'bb_upper', 'bb_lower'] and 'close' in df.columns: # Sensible default for price-like bands
                df_out[col] = df_out[col].fillna(df['close']) 
            else:
                df_out[col] = df_out[col].fillna(0) # Default for std or if close is missing

        # Price Channels (Donchian Channels)
        df_out['channel_high'] = df['high'].rolling(window=pc_window).max()
        df_out['channel_low'] = df['low'].rolling(window=pc_window).min()
        df_out['channel_mid'] = (df_out['channel_high'] + df_out['channel_low']) / 2
        # Fill initial NaNs for Price Channels
        for col in ['channel_high', 'channel_low', 'channel_mid']:
            df_out[col] = df_out[col].ffill()
            df_out[col] = df_out[col].bfill()
            if 'close' in df.columns: # Sensible default based on close price
                df_out[col] = df_out[col].fillna(df['close'])
            else:
                df_out[col] = df_out[col].fillna(0)

        return df_out
    except KeyError as e:
        logger.error(f"Error calculating Price Channels (BB, Donchian): Missing column {e}. Available: {df.columns.tolist()}")
        return pd.DataFrame(index=df.index)
    except Exception as e:
        logger.error(f"Unexpected error calculating Price Channels (BB, Donchian): {e}", exc_info=True)
        return pd.DataFrame(index=df.index)

def calculate_rsi(df: pd.DataFrame, window: int = RSI_WINDOW) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    try:
        delta = df['close'].diff().astype(float) 
        gain = (delta.where(delta > 0, 0.0)).rolling(window=window).mean() 
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean() 
        rs = gain / loss.replace(0, 1e-9) # Avoid division by zero for rs
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50) 
        return rsi.rename('rsi')
    except KeyError as e:
        logger.error(f"Error calculating RSI: Missing column {e}. Available: {df.columns.tolist()}")
        return pd.Series(np.nan, index=df.index, name='rsi')
    except Exception as e:
        logger.error(f"Unexpected error calculating RSI: {e}", exc_info=True)
        return pd.Series(np.nan, index=df.index, name='rsi')

def calculate_atr(df: pd.DataFrame, window: int = ATR_WINDOW) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    try:
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close_shifted = pd.to_numeric(df['close'].shift(), errors='coerce')

        # Construct DataFrame directly for ranges
        ranges_df = pd.DataFrame({
            'high_low': high - low,
            'high_close': np.abs(high - close_shifted),
            'low_close': np.abs(low - close_shifted)
        }, index=df.index)

        true_range = ranges_df.max(axis=1)
        atr = true_range.rolling(window=window).mean()
        atr = atr.ffill() 
        atr = atr.bfill()
        atr = atr.fillna(0) 
        return atr.rename('atr')
    except KeyError as e:
        logger.error(f"Error calculating ATR: Missing column {e}. Available: {df.columns.tolist()}")
        return pd.Series(np.nan, index=df.index, name='atr')
    except Exception as e:
        logger.error(f"Unexpected error calculating ATR: {e}", exc_info=True)
        return pd.Series(np.nan, index=df.index, name='atr')

def calculate_sma(df: pd.DataFrame, window: int) -> pd.Series:
    """Calculate Simple Moving Average (SMA)."""
    try:
        sma = df['close'].rolling(window=window).mean()
        sma = sma.ffill()
        sma = sma.bfill()
        return sma.rename(f'sma_{window}')
    except KeyError as e:
        logger.error(f"Error calculating SMA: Missing column {e}. Available: {df.columns.tolist()}")
        return pd.Series(np.nan, index=df.index, name=f'sma_{window}')
    except Exception as e:
        logger.error(f"Unexpected error calculating SMA: {e}", exc_info=True)
        return pd.Series(np.nan, index=df.index, name=f'sma_{window}')

def calculate_std_dev(df: pd.DataFrame, window: int) -> pd.Series:
    """Calculate Standard Deviation."""
    try:
        std_dev = df['close'].rolling(window=window).std()
        std_dev = std_dev.ffill()
        std_dev = std_dev.bfill()
        return std_dev.rename(f'std_dev_{window}')
    except KeyError as e:
        logger.error(f"Error calculating Std Dev: Missing column {e}. Available: {df.columns.tolist()}")
        return pd.Series(np.nan, index=df.index, name=f'std_dev_{window}')
    except Exception as e:
        logger.error(f"Unexpected error calculating Std Dev: {e}", exc_info=True)
        return pd.Series(np.nan, index=df.index, name=f'std_dev_{window}')

def calculate_price_div_sma(df: pd.DataFrame, window: int) -> pd.Series:
    """Calculate Price / SMA ratio."""
    try:
        sma = df['close'].rolling(window=window).mean()
        price_div_sma = df['close'] / sma
        price_div_sma = price_div_sma.ffill()
        price_div_sma = price_div_sma.bfill()
        return price_div_sma.rename(f'price_div_sma_{window}')
    except KeyError as e:
        logger.error(f"Error calculating Price/SMA: Missing column {e}. Available: {df.columns.tolist()}")
        return pd.Series(np.nan, index=df.index, name=f'price_div_sma_{window}')
    except Exception as e:
        logger.error(f"Unexpected error calculating Price/SMA: {e}", exc_info=True)
        return pd.Series(np.nan, index=df.index, name=f'price_div_sma_{window}')

def calculate_volatility(df: pd.DataFrame, window: int = VOLATILITY_WINDOW) -> pd.Series:
    """Calculate rolling standard deviation of returns as a volatility measure."""
    logger.info(f"[DEBUG_VOL] Entering calculate_volatility. Input df shape: {df.shape if isinstance(df, pd.DataFrame) else 'N/A'}")
    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.error("[DEBUG_VOL] Input DataFrame is invalid or empty in calculate_volatility.")
        return pd.Series(0.0, index=df.index, name='volatility')

    try:
        logger.info(f"[DEBUG_VOL] Calculating pct_change. Columns: {df.columns.tolist()}")
        if 'close' not in df.columns:
            logger.error("[DEBUG_VOL] 'close' column missing for volatility calculation.")
            return pd.Series(0.0, index=df.index, name='volatility')
            
        returns = df['close'].pct_change()
        logger.info(f"[DEBUG_VOL] pct_change calculated. Type of returns: {type(returns)}, Shape: {returns.shape if hasattr(returns, 'shape') else 'N/A'}")
        
        logger.info(f"[DEBUG_VOL] Calculating rolling std with window {window}.")
        volatility = returns.rolling(window=window).std()
        logger.info(f"[DEBUG_VOL] Rolling std calculated. Volatility series type: {type(df['volatility'])}, Shape: {df['volatility'].shape if hasattr(df['volatility'], 'shape') else 'N/A'}")
        
        volatility = volatility.ffill()
        volatility = volatility.bfill()
        volatility = volatility.fillna(0.0001) 
        logger.info("[DEBUG_VOL] Volatility NaNs handled.")
        return volatility.rename('volatility')

    except KeyError as e:
        logger.error(f"[DEBUG_VOL] KeyError calculating Volatility: Missing column {e}. Available: {df.columns.tolist()}")
        return pd.Series(0.0001, index=df.index, name='volatility')
    except Exception as e:
        logger.error(f"[DEBUG_VOL] Unexpected error calculating Volatility: {e}", exc_info=True)
        return pd.Series(0.0001, index=df.index, name='volatility')
    
    logger.info(f"[DEBUG_VOL] Exiting calculate_volatility. Output df shape: {df.shape if isinstance(df, pd.DataFrame) else 'N/A'}, Volatility NaNs: {df['volatility'].isnull().sum() if 'volatility' in df and isinstance(df, pd.DataFrame) else 'N/A'}")
    return df

def calculate_trading_ranges(df: pd.DataFrame, adx_window: int = ADX_WINDOW, tr_window: int = TRADING_RANGE_WINDOW) -> pd.DataFrame:
    """
    Calculate ADX and other range-based indicators.
    """
    if df.empty or not all(col in df.columns for col in ['high', 'low', 'close']):
        logger.warning("Input DataFrame for trading ranges is empty or missing HLC columns.")
        return pd.DataFrame()

    df_out = pd.DataFrame(index=df.index)

    # Ensure columns are float
    try:
        high_prices = df['high'].astype(float).to_numpy()
        low_prices = df['low'].astype(float).to_numpy()
        close_prices = df['close'].astype(float).to_numpy()
    except ValueError as ve:
        logger.error(f"Trading Ranges: Could not convert HLC to float: {ve}. Filling with NaN.")
        for col in ['adx', 'price_range_period']:
            df_out[col] = np.nan
        return df_out

    # ADX
    try:
        df_out['adx'] = ta.ADX(high_prices, low_prices, close_prices, timeperiod=adx_window)
    except Exception as e:
        logger.warning(f"Could not calculate ADX: {e}. Filling 'adx' with NaN.")
        df_out['adx'] = np.nan

    # Custom Price Range
    try:
        df_out['price_range_period'] = (df['high'].rolling(window=tr_window).max() - df['low'].rolling(window=tr_window).min())
    except Exception as e:
        logger.warning(f"Could not calculate custom price range: {e}. Filling with NaN.")
        df_out['price_range_period'] = np.nan

    return df_out

def calculate_technical_indicators(df: pd.DataFrame, window_size: int = 20) -> pd.DataFrame:
    """
    Calculates a suite of technical indicators and adds them to the DataFrame.
    """
    if df.empty:
        return pd.DataFrame()

    result_df = df.copy()

    indicator_functions = [
        (calculate_rsi, {'window': RSI_WINDOW}),
        (calculate_atr, {'window': ATR_WINDOW}),
        (calculate_volatility, {'window': VOLATILITY_WINDOW}),
        (calculate_vwap, {}),
        (calculate_momentum_indicators, {'window_size': 14}),
        (calculate_price_channels, {}),
        (calculate_trading_ranges, {'adx_window': ADX_WINDOW, 'tr_window': TRADING_RANGE_WINDOW})
    ]

    for func, params in indicator_functions:
        indicator_output = func(result_df, **params)
        if isinstance(indicator_output, pd.Series):
            result_df[indicator_output.name] = indicator_output
        elif isinstance(indicator_output, pd.DataFrame):
            result_df = result_df.join(indicator_output, how='left')

    # Final check for NaNs and fill them
    result_df = result_df.ffill().bfill().fillna(0)
    
    return result_df