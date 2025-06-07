# Standard library imports
from typing import (
    Tuple, Optional, List, Dict, Any, Union, Type, TypeVar, cast,
    Callable, Protocol, Sized, runtime_checkable, overload, TypeGuard
)
from datetime import timedelta
import os
import traceback
import gc
import logging

# Import logger first to ensure it's available for other imports
from logger_setup import logger, setup_logger

# Re-initialize logger with custom settings
logger = setup_logger()

# Third-party imports
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from numpy.typing import NDArray

# Import technical indicators
from technical_indicators import (
    calculate_vwap,
    calculate_rsi,
    calculate_atr,
    calculate_volatility,
    calculate_momentum_indicators,
    calculate_price_channels,
    calculate_trading_ranges
)

# Import Constants
from config_vals import (
    CURRENT_UTC_TIME, CURRENT_USER,
    UNIVERSE_FILE, RAW_DIR,
    START_CAP, MAX_DRAWDOWN, RISK_PER_TRADE, STOP_LOSS, TRANSACTION_COST, MAX_POSITION_SIZE,
    TRAILING_STOP_PERCENT, SCALE_IN_LEVELS, SCALE_OUT_LEVELS, MAX_POSITION_DURATION, MIN_LIQUIDITY_THRESHOLD,
    INPUT_DIM, SEQUENCE_LENGTH, WINDOW_SIZE, PREDICTION_WINDOW, EVAL_DAYS, VOLATILITY_ADJUSTMENT,
    BATCH_SIZE, LEARNING_RATE, GAMMA, TAU, ENTROPY_COEF, VF_COEF, MAX_GRAD_NORM,
    N_STEPS, N_EPOCHS, N_ENVS, REWARD_SCALE,
    PROGRESS_INTERVAL
)

# Type variables
T = TypeVar('T')
PandasNumericType = Union[pd.Series, pd.DataFrame]
NumericArray = Union[np.ndarray, List[float]]

@runtime_checkable
class HasLen(Protocol):
    def __len__(self) -> int: ...

def is_numeric_array(obj: Any) -> TypeGuard[Union[pd.Series, np.ndarray]]:
    """Type guard for numeric arrays."""
    return isinstance(obj, (pd.Series, np.ndarray))

def create_buckets(price_changes: Union[pd.Series, np.ndarray], is_loss: bool = True) -> Optional[List[float]]:
    """Helper function to create buckets for price changes"""
    if not is_numeric_array(price_changes) or not hasattr(price_changes, '__len__') or len(price_changes) < 2:
        return None
        
    try:
        # Convert to numpy array and filter out inf and nan values
        if isinstance(price_changes, pd.Series):
            # Ensure we get a numpy array from a Series
            price_changes_values = price_changes.to_numpy(dtype=np.float32, na_value=np.nan)
        else:
            # If it's already an ndarray (or compatible), ensure it's float32
            price_changes_values = np.asarray(price_changes, dtype=np.float32)
            
        price_changes_filtered = price_changes_values[np.isfinite(price_changes_values)]
        
        if len(price_changes_filtered) < 2:  # Check again after filtering
            return None
            
        # Get min and max values with some padding
        min_val, max_val = float(price_changes_filtered.min()), float(price_changes_filtered.max())
        padding = max(0.0001, (max_val - min_val) * 0.1)  # 10% padding
        min_val -= padding
        max_val += padding
        
        # Check if all values are the same or very close
        if np.isclose(min_val, max_val) or max_val - min_val < 1e-6:
            # Create artificial range with small differences
            step = 0.0001 if abs(min_val) < 0.01 else abs(min_val) * 0.01
            bins = [min_val - step, min_val, max_val, max_val + 2*step]  # 4 points for 3 bins
            print(f"[INFO] Created artificial bins for {'loss' if is_loss else 'profit'} with step {step}: {bins}")
        else:
            # Create 3 equal-width bins between min and max
            bins = [min_val + i * (max_val - min_val) / 3 for i in range(4)]  # 4 points for 3 bins
            print(f"[INFO] Created equal-width bins for {'loss' if is_loss else 'profit'}: {bins}")
            
        # Ensure bins are strictly increasing and unique
        bins = sorted(list(set(bins)))  # Remove duplicates and sort
        if len(bins) < 2:  # If we don't have enough unique points
            return None
            
        return bins
        
    except Exception as e:
        print(f"[WARNING] Error in create_buckets: {str(e)}")
        return None

def preprocess_data(df: pd.DataFrame, input_dim: int = INPUT_DIM, sequence_length: int = SEQUENCE_LENGTH) -> Tuple[Optional[NDArray], Optional[List[NDArray]], Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], Optional[pd.DataFrame], Optional[RobustScaler]]:
    """
    Enhanced preprocessing pipeline with robust NaN handling and data quality checks.
    
    Args:
        df: Input DataFrame with financial time series data
        input_dim: Input dimension for features
        sequence_length: Length of sequence for time series
        
    Returns:
        tuple: (X, ys, weights, df, scaler) or (None, None, None, df, None) on failure
    """
    # Make a working copy to avoid modifying the original
    df = df.copy()
    
    try:
        # 1. Initial data validation
        if df.empty:
            logger.error("Empty DataFrame provided to preprocess_data")
            return None, None, None, df, None
            
        # 2. Initial NaN check and logging
        initial_nan_cols = df.columns[df.isnull().any()].tolist()
        if initial_nan_cols:
            logger.warning(f"Initial NaN detected in columns: {initial_nan_cols}")
            logger.debug(f"NaN counts:\n{df[initial_nan_cols].isnull().sum()}")
        
        # 3. Handle missing values with forward/backward fill
        df = df.ffill().bfill()
        
        # 4. Safe calculation wrapper for technical indicators
        def safe_calculate(calc_func: Callable[..., PandasNumericType], df: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
            try:
                # Pass through df and any other relevant args/kwargs to the specific indicator function
                result = calc_func(df, *args, **kwargs)
                if isinstance(result, pd.Series):
                    result = result.to_frame()
                # Ensure all columns in the result are filled, not just the first one if it's a multi-column indicator
                for col in result.columns:
                    if pd.api.types.is_numeric_dtype(result[col]):
                        result[col] = result[col].ffill().bfill().fillna(0)
                    else:
                        # For non-numeric columns, attempt ffill/bfill, but don't fillna with 0
                        result[col] = result[col].ffill().bfill()
                return result
            except Exception as e:
                logger.error(f"Error in {calc_func.__name__}: {str(e)}")
                # Return a DataFrame with a default column if an error occurs
                # to maintain consistency in the preprocessing pipeline.
                # The name 'default' is arbitrary and signifies an error state.
                if isinstance(df, pd.DataFrame):
                    return pd.DataFrame(0, index=df.index, columns=['default'])
                # This case should ideally not be reached if df is always a DataFrame,
                # but as a fallback, create a Series then convert to DataFrame.
                return pd.Series(0, index=df.index, name='default').to_frame()
        
        # 5. Calculate technical indicators with error handling
        # Note: Some functions might require additional arguments (e.g., window size)
        # These should be passed here if they are not using default values from their definitions.
        df = safe_calculate(calculate_vwap, df)
        df = safe_calculate(calculate_rsi, df) # Assuming calculate_rsi might take a window, passed via *args or **kwargs if needed
        df = safe_calculate(calculate_atr, df) # Same for calculate_atr
        vol_df = safe_calculate(calculate_volatility, df) # Same for calculate_volatility
        
        # 6. Enhanced Winsorization with bounds checking
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        time_cols = ['timestamp', 'hour', 'minute', 'second', 'day_of_week']
        
        for col in numeric_cols:
            if col in time_cols:
                continue
                
            try:
                # Calculate percentiles safely
                valid_vals = df[col].dropna()
                if len(valid_vals) < 2:  # Need at least 2 values for percentiles
                    continue
                    
                lower, upper = np.percentile(valid_vals, [1, 99])
                if np.isnan(lower) or np.isnan(upper) or lower == upper:
                    continue
                    
                # Apply Winsorization with clipping
                df[col] = np.clip(df[col], lower, upper)
                
            except Exception as e:
                logger.warning(f"Winsorization failed for {col}: {str(e)}")
                continue
        
        # 7. Robust feature scaling
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if 'timestamp' in numeric_cols:
                numeric_cols = numeric_cols.drop('timestamp')
                
            # Handle constant columns before scaling
            for col in numeric_cols:
                if df[col].nunique() == 1:
                    df[col] = 0  # Set to zero for constant columns
                    continue
                    
            # Apply RobustScaler with clipping
            scaler = RobustScaler()
            df_scaled = pd.DataFrame(
                scaler.fit_transform(df[numeric_cols]),
                columns=numeric_cols,
                index=df.index
            )
            
            # Clip extreme values that might have been introduced by scaling
            df_scaled = df_scaled.clip(-3, 3)  # 3 standard deviations
            df[numeric_cols] = df_scaled
            
        except Exception as e:
            logger.error(f"Feature scaling failed: {str(e)}")
            return None, None, None, df, None
        
        # 8. Add volatility features if they exist
        if vol_df is not None and not vol_df.empty:
            df = pd.concat([df, vol_df], axis=1)
            logger.debug(f"Added {len(vol_df.columns)} volatility features")
        
        # 9. Ensure 'volatility' column exists for RL pipeline compatibility
        if 'close_volatility' in df.columns:
            df['volatility'] = df['close_volatility']
        elif 'atr_volatility' in df.columns:
            df['volatility'] = df['atr_volatility']
        elif 'close' in df.columns:
            logger.warning("No volatility column found, creating synthetic volatility")
            df['volatility'] = df['close'].pct_change().rolling(20).std().fillna(0.001)
        
        # 10. Ensure required columns are present
        required_cols = ['vwap', 'rsi', 'atr']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Creating synthetic versions of missing columns: {missing_cols}")
            # Create synthetic versions of missing columns where possible
            if 'vwap' in missing_cols and all(x in df.columns for x in ['close', 'volume']):
                df['vwap'] = df['close']  # Simplified approximation
            if 'rsi' in missing_cols and 'close' in df.columns:
                df['rsi'] = 50  # Neutral default
            if 'atr' in missing_cols and all(x in df.columns for x in ['high', 'low', 'close']):
                df['atr'] = (df['high'] - df['low']).rolling(14).mean().fillna((df['high'] - df['low']).mean())
        
        # 11. Final validation of required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns after synthetic creation: {missing_cols}")
            logger.debug(f"Available columns: {list(df.columns)}")
            return None, None, None, df, None
        
        # 12. Final cleanup and validation
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        if df.empty:
            logger.error("DataFrame is empty after dropping NaNs in required columns")
            return None, None, None, df, None
        
        # 13. Calculate price change with better error handling
        if all(x in df.columns for x in ['close', 'open']):
            df['price_change'] = (df.close - df.open) / df.open.replace(0, np.nan)  # Avoid division by zero
            df['price_change'] = df['price_change'].fillna(0)
        else:
            logger.warning("Missing 'close' or 'open' columns, using zero price change")
            df['price_change'] = 0.0
            
        # 14. Final NaN check
        if df.isnull().any().any():
            logger.error("NaN values still present after preprocessing")
            return None, None, None, df, None
            
    except Exception as e:
        logger.error(f"Unexpected error in preprocess_data: {str(e)}", exc_info=True)
        return None, None, None, df, None
    
    # Prepare features for model input
    try:
        # Use only numeric columns that exist in the dataframe
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        features = df[numeric_cols].values.astype(np.float32)
        logger.info(f"Prepared feature matrix with shape: {features.shape}")
        
        # Ensure features have the correct shape for the model
        if features.shape[1] < input_dim:
            logger.warning(f"Feature dimension {features.shape[1]} is less than required {input_dim}, padding with zeros")
            padding = np.zeros((features.shape[0], input_dim - features.shape[1]), dtype=np.float32)
            features = np.hstack([features, padding])
    except Exception as e:
        logger.error(f"Feature scaling failed: {str(e)}")
        logger.debug(f"Available columns: {list(df.columns)}")
        return None, None, None, df, None

    # Ensure features have the correct shape
    if features.shape[1] != input_dim:
        logger.warning(f"Feature dimension mismatch. Expected {input_dim}, got {features.shape[1]}. Adjusting...")
        # Pad or truncate features to match input_dim
        if features.shape[1] < input_dim:
            padding = np.zeros((features.shape[0], input_dim - features.shape[1]), dtype=np.float32)
            features = np.hstack([features, padding])
        else:
            features = features[:, :input_dim]

    # Head1: 0 = loss or flat, 1 = profit
    df['h1'] = (df.price_change > 0).astype(np.int32)
    
    # Head2: bucket loss only, else -1
    loss_mask = df.price_change < 0
    loss_changes = df.loc[loss_mask, 'price_change']
    
    if len(loss_changes) >= 3:  # Need at least 3 points for 4 quantiles
        bins_l = create_buckets(loss_changes, is_loss=True)
        if bins_l is not None:
            try:
                df['h2'] = pd.cut(df.price_change, bins=bins_l, labels=[0, 1, 2], include_lowest=True)
                df['h2'] = df['h2'].cat.add_categories([-1])
                df.loc[~loss_mask, 'h2'] = -1
            except Exception as e:
                print(f"[WARNING] Error in loss bucket assignment: {str(e)}")
                df['h2'] = -1
        else:
            df['h2'] = -1
    else:
        df['h2'] = -1
    
    # Ensure h2 is categorical with all possible categories
    df['h2'] = pd.Categorical(df['h2'], categories=[-1, 0, 1, 2])
    
    # Head3: bucket profit only, else -1
    prof_mask = df.price_change > 0
    prof_changes = df.loc[prof_mask, 'price_change']
    
    if len(prof_changes) >= 3:  # Need at least 3 points for 4 quantiles
        bins_p = create_buckets(prof_changes, is_loss=False)
        if bins_p is not None:
            try:
                df['h3'] = pd.cut(df.price_change, bins=bins_p, labels=[0, 1, 2], include_lowest=True)
                df['h3'] = df['h3'].cat.add_categories([-1])
                df.loc[~prof_mask, 'h3'] = -1
            except Exception as e:
                print(f"[WARNING] Error in profit bucket assignment: {str(e)}")
                df['h3'] = -1
        else:
            df['h3'] = -1
    else:
        df['h3'] = -1
    
    # Ensure h3 is categorical with all possible categories
    df['h3'] = pd.Categorical(df['h3'], categories=[-1, 0, 1, 2])

    # Build sequences and labels
    X: List[np.ndarray] = []
    ys: List[List[int]] = [[], [], []]
    
    for i in range(sequence_length, len(df)):
        try:
            # Skip if any of the labels are NaN or not finite
            if (pd.isna(df.h1.iloc[i]) or pd.isna(df.h2.iloc[i]) or pd.isna(df.h3.iloc[i]) or
                not np.isfinite(df.h1.iloc[i]) or not np.isfinite(df.h2.iloc[i]) or not np.isfinite(df.h3.iloc[i])):
                logger.warning(f"Skipping row {i} due to NaN or non-finite values in labels")
                continue
                
            # Convert labels to integers with validation
            h1 = int(round(float(df.h1.iloc[i])))
            h2 = int(round(float(df.h2.iloc[i])))
            h3 = int(round(float(df.h3.iloc[i])))
            
            # Validate the converted values
            if not (0 <= h1 <= 1):  # h1 should be binary (0 or 1)
                logger.warning(f"Invalid h1 value {h1} at row {i}, expected 0 or 1")
                continue
                
            if not (-1 <= h2 <= 2):  # h2 should be -1, 0, 1, or 2
                logger.warning(f"Invalid h2 value {h2} at row {i}, expected -1, 0, 1, or 2")
                continue
                
            if not (-1 <= h3 <= 2):  # h3 should be -1, 0, 1, or 2
                logger.warning(f"Invalid h3 value {h3} at row {i}, expected -1, 0, 1, or 2")
                continue
                
            X.append(features[i-sequence_length:i])
            ys[0].append(h1)
            ys[1].append(h2)
            ys[2].append(h3)
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error processing row {i}: {e}")
            continue
    
    # Check if we have any valid data points
    if not X:
        raise ValueError("No valid data points found after filtering NaN values")
        
    X_array = np.array(X, dtype=np.float32)
    ys_arrays = [np.array(y, dtype=np.int32) for y in ys if y]

    # Calculate class weights with safety checks for missing classes
    def safe_weights(y: np.ndarray, num_classes: int, offset: int = 0) -> torch.Tensor:
        # Shift values to handle -1 category
        y_shifted = y + offset
        # Get counts for each class, including zeros for missing classes
        counts = np.zeros(num_classes, dtype=np.float32)
        unique, counts_ = np.unique(y_shifted, return_counts=True)
        for u, c in zip(unique, counts_):
            if 0 <= u < num_classes:  # Only include valid class indices
                counts[int(u)] = c
        # Avoid division by zero and handle missing classes
        weights = 1.0 / (counts + 1e-6)
        # If all counts are zero, use uniform weights
        if np.all(counts == 0):
            weights = np.ones_like(weights, dtype=np.float32)
        return torch.tensor(weights, dtype=torch.float32)

    # Calculate weights for each head
    w1 = safe_weights(ys_arrays[0], num_classes=2)  # Binary classification (0,1)
    w2 = safe_weights(ys_arrays[1], num_classes=4, offset=1)  # 3 classes + 1 for -1
    w3 = safe_weights(ys_arrays[2], num_classes=4, offset=1)  # 3 classes + 1 for -1

    # Normalize weights
    w1 = w1 / (w1.sum() + 1e-6)
    w2 = w2 / (w2.sum() + 1e-6)
    w3 = w3 / (w3.sum() + 1e-6)

    print(f"[DEBUG-RETURN] Successful return from preprocess_data. X shape: {X_array.shape if hasattr(X_array, 'shape') else type(X_array)}, ys lens: {[len(y) for y in ys_arrays]}, df shape: {df.shape}")
    return X_array, ys_arrays, (w1, w2, w3), df, scaler
    
def clean_and_validate_dataframe(df: pd.DataFrame, symbol: str = '') -> Optional[pd.DataFrame]:
    """
    Cleans and validates the input DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("Input is not a pandas DataFrame.")
        return None
    
    if df.empty:
        logger.error("Input DataFrame is empty.")
        return None

    logger.info(f"[DEBUG_CVDF] Initial shape: {df.shape}, Initial columns: {df.columns.tolist()}")

    # Make a copy to avoid side effects
    df = df.copy()

    # Check for 'timestamp' column and make it the index
    if 'timestamp' not in df.columns:
        logger.error("`timestamp` column not found.")
        return None
        
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    logger.info(f"[DEBUG_CVDF] After to_datetime, NaNs in timestamp: {df['timestamp'].isnull().sum()}")

    # Set and sort by timestamp index
    df = df.set_index('timestamp').sort_index()
    logger.info(f"[DEBUG_CVDF] After set_index and sort_index, shape: {df.shape}")
    
    # Final check for NaN values after all processing
    if df.isnull().values.any():
        logger.error(f"[DATA VALIDATION] {symbol}: Found {df.isnull().sum().sum()} NaN values after all processing")
        return None
    
    return df

def load_and_preprocess_data():
    """
    Load and preprocess data from CSV files with enhanced validation and cleaning
    
    Returns:
        DataFrame with cleaned and validated data or None if loading fails
    """
    all_data = []
    
    # Read symbols from universe file
    logger.info("[DEBUG_LPREPROC_UNI] Entering universe file loading block.")
    try:
        logger.info(f"[DEBUG_LPREPROC_UNI] Checking existence of UNIVERSE_FILE: {UNIVERSE_FILE}")
        if not os.path.exists(UNIVERSE_FILE):
            logger.error(f"Universe file not found: {UNIVERSE_FILE}")
            return None
        logger.info("[DEBUG_LPREPROC_UNI] UNIVERSE_FILE exists. Attempting to read with pd.read_csv.")
            
        universe_df = pd.read_csv(UNIVERSE_FILE)
        logger.info(f"[DEBUG_LPREPROC_UNI] Successfully read UNIVERSE_FILE. Shape: {universe_df.shape if isinstance(universe_df, pd.DataFrame) else 'N/A'}, Columns: {universe_df.columns.tolist() if isinstance(universe_df, pd.DataFrame) else 'N/A'}")
        
        if 'symbol' not in universe_df.columns:
            logger.error("Missing 'symbol' column in universe file")
            return None
        logger.info("[DEBUG_LPREPROC_UNI] 'symbol' column found. Attempting to extract symbols.")
            
        symbols = universe_df['symbol'].dropna().tolist()
        logger.info(f"[DEBUG_LPREPROC_UNI] Successfully extracted symbols. Count: {len(symbols)}")
        
        if not symbols:
            logger.error("No valid symbols found in universe file")
            return None
            
        logger.info(f"Found {len(symbols)} symbols in universe file")
        
    except Exception as e:
        logger.error(f"Failed to load universe file: {str(e)}")
        logger.error(traceback.format_exc())
        return None

    processed_count = 0
    for sym in symbols:
        file_path = f"raw_data_a/{sym}_1min.csv"
        logger.info(f"Processing {file_path}...")
        
        try:
            # Load and validate data in chunks
            chunk_data = []
            chunks_processed = 0
            
            for chunk in load_data_in_chunks(file_path):
                if chunk is not None and not chunk.empty:
                    # Ensure timestamp is datetime and sort
                    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], errors='coerce')
                    chunk = chunk.dropna(subset=['timestamp'])
                    
                    if not chunk.empty:
                        chunk = chunk.sort_values('timestamp')
                        chunk_data.append(chunk)
                        chunks_processed += 1
            
            # Combine chunks if any were processed
            if chunk_data:
                df = pd.concat(chunk_data, ignore_index=True)
                df = df.sort_values('timestamp').drop_duplicates('timestamp')
                
                # Add symbol column if it doesn't exist
                if 'symbol' not in df.columns:
                    df['symbol'] = sym
                    
                all_data.append(df)
                processed_count += 1
                logger.info(f"Processed {len(df)} rows for {sym}")
            else:
                logger.warning(f"No valid data found for {sym}")
            
            # Clear memory
            del chunk_data
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error processing {sym}: {str(e)}")
            logger.error(traceback.format_exc())
            continue

    if not all_data:
        logger.error("No valid data loaded for any symbol")
        return None
    
    logger.info(f"Successfully processed data for {processed_count}/{len(symbols)} symbols")
    
    try:
        logger.info("[DEBUG_LPREPROC] Attempting to concatenate all_data.")
        if not all_data: # Should have been caught earlier, but double check
            logger.error("[DEBUG_LPREPROC] all_data is empty before concatenation. This shouldn't happen if processing was successful.")
            return None
        
        logger.info(f"[DEBUG_LPREPROC] Number of DataFrames in all_data: {len(all_data)}")
        if all_data:
            logger.info(f"[DEBUG_LPREPROC] Shape of first DataFrame in all_data: {all_data[0].shape if isinstance(all_data[0], pd.DataFrame) else 'Not a DataFrame'}")
            logger.info(f"[DEBUG_LPREPROC] Columns of first DataFrame: {all_data[0].columns.tolist() if isinstance(all_data[0], pd.DataFrame) else 'N/A'}")
            logger.info(f"[DEBUG_LPREPROC] Dtypes of first DataFrame: \n{all_data[0].dtypes if isinstance(all_data[0], pd.DataFrame) else 'N/A'}")


        final_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"[DEBUG_LPREPROC] Successfully concatenated all_data. Shape of final_df: {final_df.shape}")
        
        # Ensure proper data types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        logger.info(f"[DEBUG_LPREPROC] Attempting to convert numeric columns: {numeric_cols}")
        for col in numeric_cols:
            if col in final_df.columns:
                logger.info(f"[DEBUG_LPREPROC] Converting column '{col}' to numeric.")
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
                logger.info(f"[DEBUG_LPREPROC] Column '{col}' converted. NaN count: {final_df[col].isnull().sum()}")
            else:
                logger.warning(f"[DEBUG_LPREPROC] Column '{col}' not found in final_df for numeric conversion.")
        logger.info("[DEBUG_LPREPROC] Numeric column conversion loop finished.")
        
        # Final validation
        logger.info("[DEBUG_LPREPROC] Attempting final NaN check and drop.")
        if final_df.isnull().any().any():
            null_counts = final_df.isnull().sum()
            logger.warning(f"[DEBUG_LPREPROC] Found {final_df.isnull().sum().sum()} NaN values in final data before drop. "
                         f"Null counts by column: {null_counts[null_counts > 0].to_dict()}")
            
            initial_rows = len(final_df)
            final_df = final_df.dropna(subset=numeric_cols + ['timestamp'])
            if len(final_df) < initial_rows:
                logger.warning(f"[DEBUG_LPREPROC] Dropped {initial_rows - len(final_df)} rows with missing values in final data.")
        else:
            logger.info("[DEBUG_LPREPROC] No NaNs found in final_df during initial check.")
        logger.info("[DEBUG_LPREPROC] Final NaN check and drop finished.")
        
        logger.info(f"Final dataset shape: {final_df.shape}")
        return final_df
        
    except Exception as e:
        logger.error(f"Error concatenating final dataset: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def split_data(df):
    """Split data into training and validation sets"""
    max_date = df['timestamp'].max()
    train_end = max_date - timedelta(days=EVAL_DAYS)
    
    train_data = df[df['timestamp'] < train_end].copy()
    val_data = df[df['timestamp'] >= train_end].copy()
    
    return train_data, val_data

def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets respecting time order
    
    Args:
        df: DataFrame with timestamp column
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Invalid input DataFrame")
    
    df_before_split = df.copy()
    
    # Ensure the index is a DatetimeIndex for time-based splitting
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("DataFrame index must be a DatetimeIndex for split_dataset.")
        # Return empty DataFrames to prevent downstream errors
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    logger.info(f"Splitting data from {df.index.min()} to {df.index.max()}")

    # Define split points
    val_start = df.index.max() - pd.DateOffset(days=180) # Last ~6 months for validation
    test_start = df.index.max() - pd.DateOffset(days=90)  # Last ~3 months for test

    # Perform splits based on index
    train_df = df[df.index < val_start]
    val_df = df[(df.index >= val_start) & (df.index < test_start)]
    test_df = df[df.index >= test_start]

    # Logging results
    logger.info(f"Train period: {train_df.index.min()} to {train_df.index.max()}")
    logger.info(f"Validation period: {val_df.index.min()} to {val_df.index.max()}")
    logger.info(f"Test period: {test_df.index.min()} to {test_df.index.max()}")
    logger.info(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")

    return train_df, val_df, test_df

def load_data_in_chunks(file_path, chunk_size=10000):
    """
    Load data in chunks to reduce memory usage with enhanced validation
    
    Args:
        file_path: Path to the CSV file
        chunk_size: Number of rows per chunk
        
    Yields:
        DataFrame chunks with validated data
    """
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    try:
        # First, validate the file exists and is readable
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
            
        # Read the first row to check columns
        with open(file_path, 'r') as f:
            first_line = f.readline().strip().lower()
            if not all(col in first_line for col in required_columns):
                logger.error(f"Missing required columns in {file_path}")
                return None
    except Exception as e:
        logger.error(f"Error validating file {file_path}: {str(e)}")
        return None
    
    try:
        # Process file in chunks with enhanced validation
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, 
                               parse_dates=['timestamp'],
                               infer_datetime_format=True):
            if chunk.empty:
                continue
                
            # Ensure required columns exist and have correct types
            chunk = chunk.copy()
            for col in required_columns:
                if col not in chunk.columns:
                    logger.error(f"Missing required column: {col}")
                    continue
                
                # Convert numeric columns
                if col != 'timestamp':
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
                    
                    # Handle non-positive prices
                    if col in ['open', 'high', 'low', 'close']:
                        mask = (chunk[col] <= 0) | chunk[col].isna()
                        if mask.any():
                            logger.warning(f"Found {mask.sum()} invalid/zero prices in {file_path}")
                            chunk[col] = chunk[col].replace(0, np.nan)
                    
                    # Handle volume
                    elif col == 'volume':
                        chunk[col] = chunk[col].clip(lower=0)
            
            # Drop rows with any remaining NaNs in required columns
            initial_rows = len(chunk)
            chunk = chunk.dropna(subset=required_columns)
            
            if len(chunk) < initial_rows:
                logger.warning(f"Dropped {initial_rows - len(chunk)} rows with missing values in {file_path}")
                
            if not chunk.empty:
                yield chunk
                
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def safe_process_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Safely process numeric columns with proper type checking."""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        time_cols = ['timestamp', 'hour', 'minute', 'second', 'day_of_week']
        
        for col in numeric_cols:
            if col in time_cols:
                continue
                
            series = df[col]
            if not isinstance(series, pd.Series):
                logger.warning(f"Column {col} is not a pandas Series, skipping")
                continue
                
            # Check for unique values
            unique_values = series.value_counts()
            if len(unique_values) <= 1:
                continue
                
            # Process valid values
            valid_vals = series.dropna()
            if len(valid_vals) < 2:
                continue
                
            try:
                # Convert to numpy array explicitly
                np_vals = np.array(valid_vals.values, dtype=np.float64)
                lower = float(np.percentile(np_vals, 1))
                upper = float(np.percentile(np_vals, 99))
                
                if np.isfinite(lower) and np.isfinite(upper) and lower != upper:
                    df[col] = series.clip(lower=lower, upper=upper)
            except Exception as e:
                logger.warning(f"Error processing column {col}: {str(e)}")
                
        return df
        
    except Exception as e:
        logger.error(f"Error in safe_process_numeric_columns: {str(e)}")
        return df

def calculate_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volatility features with proper type handling."""
    try:
        if 'close' not in df.columns:
            return pd.DataFrame()
            
        close_series = df['close']
        if not isinstance(close_series, pd.Series):
            return pd.DataFrame()
            
        # Calculate returns
        returns = close_series.pct_change()
        
        # Calculate volatility features
        result = pd.DataFrame(index=df.index)
        result['close_volatility'] = returns.rolling(window=20).std().fillna(0.001)
        result['atr_volatility'] = df['atr'] if 'atr' in df.columns else result['close_volatility']
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating volatility features: {str(e)}")
        return pd.DataFrame()

def apply_feature_scaling(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[RobustScaler]]:
    """Apply feature scaling with proper type handling."""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop('timestamp') if 'timestamp' in numeric_cols else numeric_cols
        
        # Remove constant columns
        cols_to_scale = []
        for col in numeric_cols:
            series = df[col]
            if isinstance(series, pd.Series):
                # Convert to numpy array for checking unique values
                np_vals = np.array(series.dropna().values, dtype=np.float64)
                if len(np.unique(np_vals)) > 1:
                    cols_to_scale.append(col)
                else:
                    df[col] = 0
        
        if not cols_to_scale:
            return df, None
            
        # Apply scaling
        scaler = RobustScaler()
        data_to_scale = df[cols_to_scale].values.astype(np.float64)
        scaled_values = scaler.fit_transform(data_to_scale)
        
        # Convert to DataFrame and clip values
        scaled_df = pd.DataFrame(
            np.clip(scaled_values, -3, 3),
            columns=cols_to_scale,
            index=df.index
        )
        
        df[cols_to_scale] = scaled_df
        return df, scaler
        
    except Exception as e:
        logger.error(f"Error in feature scaling: {str(e)}")
        return df, None

def generate_sl_labels(df: pd.DataFrame, future_steps: int = 5, threshold_pct: float = 0.005) -> pd.DataFrame:
    """
    Generates supervised learning action labels based on future price movements.
    Uses max future price for BUYs and min future price for SELLs within the look_forward window.

    Args:
        df (pd.DataFrame): DataFrame with at least 'close' prices.
        future_steps (int): How many steps into the future to look for a price change.
        threshold_pct (float): Percentage change required to trigger a BUY or SELL signal.

    Returns:
        pd.DataFrame: The input DataFrame with an added 'sl_action_label' column.
                      0: HOLD, 1: BUY, 2: SELL
    """
    buy_threshold_pct = threshold_pct
    # SELLs require a price drop and a MACD bearish crossover/strong weakening
    sell_price_threshold_pct = threshold_pct * 1.5 
    logger.info(f"Generating SL labels with future_steps={future_steps}, BUY threshold_pct={buy_threshold_pct}, "
                f"SELL price_threshold_pct={sell_price_threshold_pct} AND MACD weakening/bearish crossover using future min/max logic.")
    
    required_cols_for_sell = ['macd', 'macd_signal']
    
    if 'close' not in df.columns or not all(col in df.columns for col in required_cols_for_sell):
        missing_req = [col for col in required_cols_for_sell if col not in df.columns]
        logger.error(f"DataFrame must contain 'close' and {required_cols_for_sell} columns for label generation. Missing: {missing_req}")
        raise ValueError(f"DataFrame must contain 'close' and {required_cols_for_sell}.")

    df_copy = df.copy()
    df_copy['sl_action_label'] = 0  # Default to HOLD

    # Calculate MACD histogram if not already present (it should be from calculate_momentum_indicators)
    if 'macd_hist' not in df_copy.columns:
        logger.info("[DEBUG_SL] 'macd_hist' not found, calculating from 'macd' and 'macd_signal'.")
        df_copy['macd_hist'] = df_copy['macd'] - df_copy['macd_signal']
    else:
        logger.info("[DEBUG_SL] 'macd_hist' found in DataFrame.")

    # Shifted MACD histogram for previous value
    df_copy['macd_hist_previous'] = df_copy['macd_hist'].shift(1)

    # Calculate future min and max prices over the window
    # The rolling window looks over 'future_steps' candles.
    # .shift(-future_steps + 1) aligns the start of the window correctly if we want to include current step + future_steps-1
    # However, for "price movement from current step i", we want to look at i+1 to i+future_steps.
    # So, we shift the result of rolling max/min back by `future_steps`.
    # The window for rolling should be `future_steps`.
    
    # To look from step i+1 to i+future_steps:
    # First, create a series of 'close' prices shifted by 1 to start looking from next candle
    close_shifted = df_copy['close'].shift(-1)
    
    # Then, roll over this shifted series for 'future_steps'
    df_copy['future_max_in_window'] = close_shifted.rolling(window=future_steps, min_periods=1).max()
    df_copy['future_min_in_window'] = close_shifted.rolling(window=future_steps, min_periods=1).min()

    # Shift these results back to align with the current time step 'i'
    # The value at index 'i' for future_max_in_window should be the max from i+1 to i+future_steps
    df_copy['future_max_in_window'] = df_copy['future_max_in_window'].shift(-(future_steps -1) -1) # Simpler: shift(-(future_steps)) if rolling on original and then shift
                                                                                                # Correct shift if rolling on close_shifted by 1
                                                                                                # No, this is getting complicated. Let's simplify.

    # Simpler vectorized approach:
    # For each row `i`, we need to look at rows `i+1` to `i+future_steps`.
    # Example: future_steps = 5. For row `i`, consider `i+1, i+2, i+3, i+4, i+5`.
    
    # Max price in the next 'future_steps' candles (from i+1 to i+future_steps)
    # The .rolling() window size is future_steps.
    # We then shift the result backwards so that for row `i`, it contains the future aggregate.
    df_copy['future_max'] = df_copy['close'].rolling(window=future_steps, min_periods=1).max().shift(-future_steps)
    df_copy['future_min'] = df_copy['close'].rolling(window=future_steps, min_periods=1).min().shift(-future_steps)

    # BUY condition: if future_max is 'buy_threshold_pct' greater than current_price
    buy_condition = (df_copy['future_max'] > df_copy['close'] * (1 + buy_threshold_pct)) & (df_copy['close'] > 0)
    df_copy.loc[buy_condition, 'sl_action_label'] = 1

    # SELL condition: combines price drop with MACD bearish crossover
    price_drop_condition = (df_copy['future_min'] < df_copy['close'] * (1 - sell_price_threshold_pct)) & (df_copy['close'] > 0)
    
    # MACD weakening/bearish crossover condition:
    # 1. MACD was above signal line (histogram was positive)
    # 2. Histogram is decreasing
    # 3. Histogram has fallen significantly (e.g., by 90% of its previous value or turned negative)
    macd_weakening_condition = (
        (df_copy['macd_hist_previous'] > 0) & 
        (df_copy['macd_hist'] < df_copy['macd_hist_previous']) & 
        (df_copy['macd_hist'] < (df_copy['macd_hist_previous'] * 0.1))
    )
    
    sell_condition = price_drop_condition & macd_weakening_condition
    
    # Apply SELL only if not already a BUY 
    df_copy.loc[sell_condition & (df_copy['sl_action_label'] != 1), 'sl_action_label'] = 2
    
    # Drop rows where future_max/future_min or macd_hist_previous could not be calculated
    df_copy.dropna(subset=['future_max', 'future_min', 'macd_hist_previous'], inplace=True)
    
    logger.info(f"[DEBUG_SL] Vectorized label generation finished. SL label counts: \n{df_copy['sl_action_label'].value_counts(normalize=True, dropna=False)}")
    logger.info("[DEBUG_SL] Returning from generate_sl_labels function.")
    return df_copy

if __name__ == "__main__":
    # Configure logger for standalone script execution if needed, 
    # though it's already set up at the module level.
    logger.info("--- Entered __main__ block of preprocess_data.py ---")
    logger.info("Starting standalone preprocessing for SL data generation...")

    output_dir = "data"
    output_file_name = "processed_with_sl_labels.csv"
    output_file_path = os.path.join(output_dir, output_file_name)
    logger.info(f"Output path configured: {output_file_path}")

    try:
        # Ensure the output directory exists
        logger.info(f"Attempting to create output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory '{output_dir}' ensured.")

        # 1. Load and initially preprocess data from raw files
        logger.info("Step 1: Loading and performing initial preprocessing of raw data...")
        combined_df = load_and_preprocess_data()
        logger.info(f"--- Returned from load_and_preprocess_data. combined_df is None: {combined_df is None}. Is empty: {combined_df.empty if isinstance(combined_df, pd.DataFrame) else 'N/A'} ---")

        if combined_df is None or combined_df.empty:
            logger.error("Failed to load or combine raw data. SL data generation aborted.")
        else:
            logger.info(f"Successfully loaded combined raw data. Shape: {combined_df.shape}")
            
            if isinstance(combined_df.index, pd.DatetimeIndex) and combined_df.index.name == 'timestamp':
                logger.info("Resetting DatetimeIndex 'timestamp' to a column for further processing.")
                combined_df = combined_df.reset_index()
            elif 'timestamp' not in combined_df.columns and combined_df.index.name == 'timestamp':
                 logger.info("Resetting named index 'timestamp' to a column for further processing.")
                 combined_df = combined_df.reset_index()

            logger.info("Step 2: Cleaning, validating, and calculating technical indicators...")
            validated_df = clean_and_validate_dataframe(combined_df, symbol="ALL_SYMBOLS_FOR_SL_MAIN") # Changed symbol slightly for clarity
            logger.info(f"--- Returned from clean_and_validate_dataframe. validated_df is None: {validated_df is None}. Is empty: {validated_df.empty if isinstance(validated_df, pd.DataFrame) else 'N/A'} ---")

            if validated_df is None or validated_df.empty:
                logger.error("Data validation and indicator calculation failed. SL data generation aborted.")
            else:
                logger.info(f"Successfully validated and added indicators. Shape: {validated_df.shape}")
                logger.info(f"[DEBUG_PREPROC_MAIN] Columns in validated_df: {validated_df.columns.tolist()}")

                logger.info("Step 3: Generating Supervised Learning (SL) labels...")
                df_with_labels = generate_sl_labels(validated_df) 
                logger.info(f"--- Returned from generate_sl_labels. df_with_labels is None: {df_with_labels is None}. Is empty: {df_with_labels.empty if isinstance(df_with_labels, pd.DataFrame) else 'N/A'} ---")

                if df_with_labels is not None and isinstance(df_with_labels, pd.DataFrame) and not df_with_labels.empty:
                    logger.info(f"Successfully generated SL labels. Shape: {df_with_labels.shape}. Columns: {df_with_labels.columns.tolist()}")
                    logger.info("Step 4: Saving data with SL labels...") # Shortened log
                    try:
                        if 'timestamp' in df_with_labels.columns:
                             df_with_labels.set_index('timestamp', inplace=True)
                        elif not isinstance(df_with_labels.index, pd.DatetimeIndex):
                            logger.warning("DataFrame index is not a DatetimeIndex for saving. Attempting to save as is.")

                        df_with_labels.to_csv(output_file_path)
                        logger.info(f"Successfully saved data with SL labels to '{output_file_path}'.")
                        logger.info("--- SL data preprocessing complete. --- ")
                    except Exception as e_save:
                        logger.error(f"Failed to save data with SL labels: {e_save}", exc_info=True)
                else:
                    logger.error(f"[ERROR_MAIN] generate_sl_labels returned an invalid or empty DataFrame.")
                    # Add more detailed logging about the state of df_with_labels
                    if df_with_labels is None: logger.error("[ERROR_MAIN] df_with_labels is None.")
                    elif not isinstance(df_with_labels, pd.DataFrame): logger.error(f"[ERROR_MAIN] df_with_labels is not a DataFrame. Type: {type(df_with_labels)}.")
                    elif df_with_labels.empty: logger.error("[ERROR_MAIN] df_with_labels is an empty DataFrame.")
        logger.info("--- Reached end of main processing logic in __main__ block. ---")
    except Exception as e_main_block:
        logger.error(f"CRITICAL ERROR in __main__ block of preprocess_data.py: {e_main_block}", exc_info=True)
    finally:
        logger.info("--- Exiting __main__ block of preprocess_data.py (finally). ---")
        from logger_setup import RateLimitedLogger # Import RateLimitedLogger for isinstance check
        if isinstance(logger, RateLimitedLogger) and hasattr(logger, '_flush_buffer'):
            logger.info("Flushing logger buffer in __main__ finally...")
            logger._flush_buffer()
            logger.info("Logger buffer flushed in __main__ finally.")
