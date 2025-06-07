import os
from datetime import datetime
from typing import Dict

# User and timestamp information
CURRENT_UTC_TIME = "2025-05-29 11:03:35"
CURRENT_USER = "Mah0ksha"
TIME_FILTERS = {'start_hour': 9, 'start_minute': 30, 'end_hour': 15, 'end_minute': 0}  # Trading hours


# File and directory paths
UNIVERSE_FILE = os.getenv("UNIVERSE_FILE", "universe.csv")
RAW_DIR = os.getenv("RAW_DIR", "raw_data_a")

# Capital and Risk Management
START_CAP = 5000  # Initial capital
TRANSACTION_COST = 0.001  # 0.1% transaction cost

# Adjusted Risk Management Parameters
MAX_DRAWDOWN = 0.15  # Reduced from 0.2 to 0.15 (15% maximum drawdown)
RISK_PER_TRADE = 0.01  # Reduced from 0.02 to 0.01 (1% risk per trade)
STOP_LOSS = 0.015  # Reduced from 0.02 to 0.015 (1.5% stop loss)
MAX_POSITION_SIZE = 0.05  # Reduced from 0.1 to 0.05 (5% max position)
TRAILING_STOP_PERCENT = 0.01  # Reduced from 0.02 to 0.01 (1% trailing stop)

# Position Management
SCALE_IN_LEVELS = 3  # Number of scale-in levels
SCALE_OUT_LEVELS = 3  # Number of scale-out levels
MAX_POSITION_DURATION = 1440  # Maximum position duration in minutes (24 hours)
MIN_LIQUIDITY_THRESHOLD = 1000  # Minimum volume for liquidity

# Added Risk Management Parameters
DYNAMIC_POSITION_SCALING = True
MAX_LEVERAGE = 1.0  # No leverage
MIN_POSITION_SIZE = 0.001  # Minimum position size as fraction of capital
POSITION_STEP_SIZE = 0.001  # Granular position size adjustments

# Reward calculation parameters
REWARD_SCALE = 1.0
MAX_POSITION_DURATION_HOURS = 24.0
MIN_TRADE_INTERVAL_MINUTES = 5.0
OPTIMAL_POSITION_SIZE_MIN = 0.3
OPTIMAL_POSITION_SIZE_MAX = 0.7

# Add after other constants
BASE_TIMESTEPS = 10000  # Base training timesteps
MAX_TIMESTEPS = BASE_TIMESTEPS * 2  # Maximum allowed timesteps
PERFORMANCE_THRESHOLD = -0.05  # -5% return threshold
CHUNK_SIZE = 2000  # Training chunk size

# Market Hours
MARKET_HOURS: Dict[str, int] = {
    'start_hour': 9,
    'start_minute': 30,
    'end_hour': 16,
    'end_minute': 0
}

# Data Processing
INPUT_DIM = 64  # Input dimension for the model
SEQUENCE_LENGTH = 30  # Length of sequence for LSTM/transformers
WINDOW_SIZE = 20  # Lookback window size
PREDICTION_WINDOW = 5  # Prediction window size
EVAL_DAYS = 30  # Number of days for evaluation
VOLATILITY_ADJUSTMENT = 0.5  # Volatility adjustment factor

# Model Training Parameters
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
GAMMA = 0.99  # Discount factor
TAU = 0.95  # GAE parameter
ENTROPY_COEF = 0.01  # Entropy coefficient for exploration
VF_COEF = 0.5  # Value function coefficient
MAX_GRAD_NORM = 0.5  # Clip gradients
N_STEPS = 2048  # Number of steps per update
N_EPOCHS = 10  # Number of epochs for PPO
N_ENVS = 16  # Number of parallel environments

# Training Parameters
RL_TIMESTEPS = 20000  # Increased from 10000 to 20000
EVAL_STEPS = 2000  # Added evaluation steps
WARMUP_STEPS = 1000  # Added warmup period

# Logging and Progress
PROGRESS_INTERVAL = 100  # Log progress every 100 steps