import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import argparse
from typing import Tuple

from logger_setup import setup_logger
logger = setup_logger()

# These should match the TradingEnv configuration
WINDOW_SIZE = 20 
NUM_FEATURES = 10 # price, volume, position, rsi, macd, macd_signal, bb_middle, bb_upper, bb_lower, adx
FLATTENED_OBS_DIM = WINDOW_SIZE * NUM_FEATURES

class SLPolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 3, hidden_dim: int = 256):
        super().__init__()
        # Architecture should be compatible with SB3's MlpPolicy if weights are transferred.
        # SB3 MlpPolicy typically has 2 hidden layers of 64 units by default for policy_net.
        # We can make this configurable or match a common SB3 setup.
        # For now, using a slightly larger network, can be adjusted.
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim) # Logits for 3 actions: HOLD, BUY, SELL
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3: # Should be (batch, window_size, num_features)
            x = x.view(x.size(0), -1) # Flatten to (batch, window_size * num_features)
        elif x.ndim != 2: # Expected (batch, flattened_features)
            raise ValueError(f"Input tensor expected to be 2D or 3D, got {x.ndim}D")
        return self.network(x)

class SupervisedLearningDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        # Features are expected to be already flattened: (num_samples, window_size * num_features)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def prepare_sl_data(df: pd.DataFrame, window_size: int, num_env_features: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares data for the SupervisedLearningDataset.
    Each sample in X should be one observation window, flattened.
    y should be the corresponding sl_action_label.
    This function must replicate how TradingEnv.get_current_observation() creates observations.
    """
    logger.info(f"Preparing SL data from DataFrame with {len(df)} rows. Window: {window_size}, Features: {num_env_features}")
    
    X_list = []
    y_list = []

    if 'sl_action_label' not in df.columns:
        raise ValueError("DataFrame must contain 'sl_action_label' column.")

    # Define the exact feature columns and their order as used in TradingEnv.get_current_observation
    # These are: price_obs, volume_obs, position_obs, rsi_obs, macd_obs, macd_signal_obs, 
    #            bb_middle_obs, bb_upper_obs, bb_lower_obs, adx_obs
    # The source columns in the DataFrame should be (for indicators, these are direct names):
    # 'close' (for price_obs)
    # 'volume' (for volume_obs)
    # (position_obs is generated as zeros)
    # 'rsi', 'macd', 'macd_signal', 'bb_middle', 'bb_upper', 'bb_lower', 'adx'
    
    indicator_feature_names = ['rsi', 'macd', 'macd_signal', 'bb_middle', 'bb_upper', 'bb_lower', 'adx']
    base_feature_names = ['close', 'volume']
    all_required_df_cols = base_feature_names + indicator_feature_names + ['sl_action_label']

    missing_cols = [col for col in all_required_df_cols if col not in df.columns and col != 'sl_action_label']
    if missing_cols:
        raise ValueError(f"Missing required columns in DataFrame for SL data prep: {missing_cols}. Ensure preprocessing generates these.")

    df_copy = df.copy()
    
    # Fill NaNs for feature columns (important for indicators at the start of the series)
    feature_cols_to_fill = base_feature_names + indicator_feature_names
    for col in feature_cols_to_fill:
        if df_copy[col].isnull().any():
            # logger.debug(f"Column '{col}' has NaNs. Filling with ffill then bfill.")
            df_copy[col] = df_copy[col].ffill().bfill()
        if df_copy[col].isnull().any(): # If still NaNs (e.g., entire column was NaN)
            logger.warning(f"Column '{col}' still has NaNs after ffill/bfill. Filling with 0.")
            df_copy[col] = df_copy[col].fillna(0)
    
    # Iterate from the first point where a full window can be formed
    for i in range(window_size - 1, len(df_copy)):
        observation_features_for_window = []
        
        # 1. Price observations (from 'close' column)
        observation_features_for_window.append(df_copy['close'].iloc[i - window_size + 1 : i + 1].values)
        
        # 2. Volume observations
        observation_features_for_window.append(df_copy['volume'].iloc[i - window_size + 1 : i + 1].values)
        
        # 3. Position observations (neutral for SL pre-training)
        observation_features_for_window.append(np.zeros(window_size, dtype=np.float32))
        
        # 4-10. Technical indicators
        for indicator_name in indicator_feature_names:
            observation_features_for_window.append(df_copy[indicator_name].iloc[i - window_size + 1 : i + 1].values)
            
        # Stack them column-wise to get (window_size, num_env_features)
        # np.column_stack expects a list of 1D arrays or 2D arrays to be stacked as columns
        try:
            current_observation_array = np.column_stack(observation_features_for_window)
        except ValueError as e:
            logger.error(f"Error during np.column_stack at index {i}: {e}. Shapes of collected features:")
            for idx, arr in enumerate(observation_features_for_window):
                logger.error(f"Feature {idx} shape: {arr.shape if isinstance(arr, np.ndarray) else type(arr)}")
            continue # Skip this sample

        if current_observation_array.shape != (window_size, num_env_features):
            logger.error(f"Observation shape mismatch at index {i}. Expected ({window_size}, {num_env_features}), got {current_observation_array.shape}. Skipping sample.")
            continue

        X_list.append(current_observation_array.flatten()) # Flatten for MLP
        y_list.append(df_copy['sl_action_label'].iloc[i])

    if not X_list:
        logger.error("No data generated for SL training. Check DataFrame content, NaN handling, and windowing logic.")
        return np.array([]), np.array([])
        
    # Convert y_list to int64 specifically if it's not already, to avoid potential issues with torch.long
    y_array = np.array(y_list, dtype=np.int64)

    return np.array(X_list, dtype=np.float32), y_array


def evaluate_sl_model(model_path: str, val_loader: DataLoader, device: torch.device, input_dim: int, num_classes: int = 3):
    """
    Evaluates the trained SL model by loading its weights and computing detailed metrics.
    """
    logger.info("--- Executing evaluate_sl_model ---")
    logger.info(f"==== Starting SL Model Evaluation: {model_path} ====")
    
    model = SLPolicyNetwork(input_dim=input_dim, output_dim=num_classes)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Successfully loaded model weights from {model_path}")
    except FileNotFoundError:
        logger.error(f"SL model file not found at {model_path}. Cannot evaluate.")
        logger.info("--- Exiting evaluate_sl_model due to FileNotFoundError ---")
        return
    except Exception as e:
        logger.error(f"Error loading SL model from {model_path}: {e}")
        logger.info(f"--- Exiting evaluate_sl_model due to error loading model ---")
        return
        
    model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    if not all_labels or not all_predictions:
        logger.info("No labels or predictions collected during SL model evaluation. Cannot generate report.")
        logger.info("--- Exiting evaluate_sl_model due to no labels/predictions ---")
        return

    logger.info("\n==== SL Model Evaluation Report ====")
    
    target_names = ['HOLD (0)', 'BUY (1)', 'SELL (2)'] 
    try:
        cm = confusion_matrix(all_labels, all_predictions, labels=list(range(num_classes)))
        logger.info(f"Confusion Matrix:\n{cm}")

        report = classification_report(all_labels, all_predictions, target_names=target_names, labels=list(range(num_classes)), zero_division=0)
        logger.info(f"Classification Report:\n{report}")
    except Exception as e:
        logger.error(f"Error generating sklearn metrics: {e}")
        logger.info(f"Labels unique: {np.unique(all_labels)}, Predictions unique: {np.unique(all_predictions)}")

    logger.info("==== SL Model Evaluation Finished ====")
    logger.info("--- Finished evaluate_sl_model ---")


def train_sl_model(
    processed_data_path: str, 
    model_save_path: str,
    window_size: int = WINDOW_SIZE,
    num_features: int = NUM_FEATURES,
    epochs: int = 10, 
    batch_size: int = 64, 
    learning_rate: float = 1e-3,
    test_size: float = 0.1
    ):

    logger.info(f"Starting SL model training. Data: {processed_data_path}, Save: {model_save_path}")
    
    try:
        df = pd.read_csv(processed_data_path, index_col='timestamp', parse_dates=True)
    except Exception as e:
        logger.error(f"Failed to load processed data from {processed_data_path}: {e}")
        if hasattr(logger, '_flush_buffer'): # Attempt to flush if possible
            logger._flush_buffer()
        return

    if df.empty:
        logger.error(f"Loaded DataFrame from {processed_data_path} is empty.")
        if hasattr(logger, '_flush_buffer'):
            logger._flush_buffer()
        return

    try: # Main try block for the rest of the function
        X, y = prepare_sl_data(df, window_size, num_features)

        if X.shape[0] == 0 or y.shape[0] == 0 or X.shape[0] != y.shape[0]:
            logger.error(f"SL data preparation failed or resulted in no data. X shape: {X.shape}, y shape: {y.shape}")
            return # Return early, finally block will still execute
        
        logger.info(f"Prepared SL data. X shape: {X.shape}, y shape: {y.shape}")
        logger.info(f"SL label distribution in prepared data: {pd.Series(y).value_counts(normalize=True)}")

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, shuffle=False, stratify=None)
        
        if len(X_train) == 0 or len(X_val) == 0:
            logger.error(f"Train or validation set is empty after split. Original data size: {X.shape[0]}. Need more data or smaller test_size.")
            return

        # Calculate class weights for handling imbalance (using training labels y_train)
        try:
            unique_classes = np.unique(y_train)
            if len(unique_classes) < 3 and len(unique_classes) > 0: # If some classes are missing in train set after split
                 logger.warning(f"Only {len(unique_classes)} classes present in y_train. Expected 3. Weights might be skewed or training difficult.")
                 # Fallback: equal weights or handle as error depending on policy. For now, proceed but log.
                 class_weights_array = compute_class_weight('balanced', classes=unique_classes, y=y_train)
                 # Need to map these weights to a 3-element array for all possible classes 0, 1, 2
                 # This is a simplified handling. A more robust way might be needed if classes are often missing.
                 temp_weights = {cls: weight for cls, weight in zip(unique_classes, class_weights_array)}
                 class_weights_list = [temp_weights.get(i, 0.1) for i in range(3)] # Default to small weight if class missing entirely
                 class_weights_array = np.array(class_weights_list)
            elif len(unique_classes) == 0:
                logger.error("No classes present in y_train. Cannot compute class weights.")
                return
            else:
                 class_weights_array = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y_train)

            # Cap the maximum weight to prevent excessive penalization
            max_allowed_weight = 3.0  # Define a cap for the weights - Reduced from 5.0
            capped_weights_array = np.clip(class_weights_array, a_min=None, a_max=max_allowed_weight)
            logger.info(f"Original balanced class weights: {class_weights_array}")
            logger.info(f"Capped class weights (max={max_allowed_weight}): {capped_weights_array}")
            
            class_weights_tensor = torch.tensor(capped_weights_array, dtype=torch.float32)
            logger.info(f"Calculated class weights for CrossEntropyLoss: {class_weights_tensor}")
        except Exception as e_weights:
            logger.error(f"Error calculating class weights: {e_weights}. Proceeding without weights.")
            class_weights_tensor = None # Fallback to no weights

        train_dataset = SupervisedLearningDataset(X_train, y_train)
        val_dataset = SupervisedLearningDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        flattened_input_dim = window_size * num_features
        sl_model = SLPolicyNetwork(input_dim=flattened_input_dim, output_dim=3) # 3 actions: HOLD, BUY, SELL
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sl_model.to(device)

        optimizer = optim.Adam(sl_model.parameters(), lr=learning_rate)
        # Use weighted CrossEntropyLoss
        if class_weights_tensor is not None:
            class_weights_tensor = class_weights_tensor.to(device) # Move weights to the same device as model
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            logger.info("Using weighted CrossEntropyLoss.")
        else:
            criterion = nn.CrossEntropyLoss() 
            logger.info("Using standard CrossEntropyLoss (weights calculation failed or not applicable).")

        logger.info(f"SL Model architecture: {sl_model}")
        logger.info(f"Training on {device} for {epochs} epochs. Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        best_val_loss = float('inf')

        for epoch in range(epochs):
            sl_model.train()
            train_loss_sum = 0.0
            correct_train = 0
            total_train = 0
            for batch_idx, (features, labels) in enumerate(train_loader):
                features, labels = features.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = sl_model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss_sum += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            avg_train_loss = train_loss_sum / len(train_dataset) if len(train_dataset) > 0 else 0
            train_accuracy = 100 * correct_train / total_train if total_train > 0 else 0
            
            sl_model.eval()
            val_loss_sum = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = sl_model(features)
                    loss = criterion(outputs, labels)
                    val_loss_sum += loss.item() * features.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            avg_val_loss = val_loss_sum / len(val_dataset) if len(val_dataset) > 0 else 0
            val_accuracy = 100 * correct_val / total_val if total_val > 0 else 0

            logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(sl_model.state_dict(), model_save_path)
                logger.info(f"Model improved and saved to {model_save_path} (Val Loss: {best_val_loss:.4f})")
                
        logger.info(f"Supervised pre-training loop finished. Best validation loss: {best_val_loss:.4f}") # Changed log message slightly
        logger.info(f"Best SL model saved to: {model_save_path}")

        logger.info(f"Checking if model file exists for evaluation: {model_save_path}")
        if os.path.exists(model_save_path):
            logger.info(f"Model file found. Evaluating the best saved SL model on the validation set...")
            evaluate_sl_model(
                model_path=model_save_path, 
                val_loader=val_loader, 
                device=device,
                input_dim=flattened_input_dim,
                num_classes=3
            )
        else:
            logger.warning(f"Best model file {model_save_path} NOT found. Skipping evaluation.")

    except Exception as e_main:
        logger.error(f"An error occurred during SL model training: {e_main}", exc_info=True) # Log exception info
    finally:
        logger.info("--- train_sl_model function finalizing ---")
        if hasattr(logger, '_flush_buffer'):
            logger.info("Flushing logger buffer...")
            logger._flush_buffer()
            logger.info("Logger buffer flushed.")
        else:
            logger.warning("Logger does not have _flush_buffer method.")
        logger.info("--- train_sl_model function finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a supervised learning model for policy pre-training.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the processed CSV data file (with SL labels and all required indicator columns).")
    parser.add_argument("--model_save_path", type=str, required=True, help="Path to save the trained SL model weights (.pth file).")
    parser.add_argument("--window_size", type=int, default=WINDOW_SIZE, help="Observation window size.")
    parser.add_argument("--num_features", type=int, default=NUM_FEATURES, help="Number of features per step in the window (e.g., 10 for TradingEnv).")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--test_split_size", type=float, default=0.1, help="Fraction of data for validation set (using temporal split via shuffle=False).")

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        exit(1)
        
    save_dir = os.path.dirname(args.model_save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Created directory for saving SL model: {save_dir}")
    
    train_sl_model(
        processed_data_path=args.data_path,
        model_save_path=args.model_save_path,
        window_size=args.window_size,
        num_features=args.num_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        test_size=args.test_split_size
    ) 