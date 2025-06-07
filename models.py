# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

# Stable-Baselines3 imports
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import spaces, Env
import gymnasium as gym

# Utility imports
import numpy as np
import logging
import traceback
import os
from typing import List, Dict, Optional, Tuple, Any, Union

# Import logger first to ensure it's available for other imports
from logger_setup import logger

# Import Constants
from config_vals import (
    CURRENT_UTC_TIME, CURRENT_USER,
    UNIVERSE_FILE, RAW_DIR,
    START_CAP, MAX_DRAWDOWN, RISK_PER_TRADE, STOP_LOSS, TRANSACTION_COST, MAX_POSITION_SIZE,
    TRAILING_STOP_PERCENT, SCALE_IN_LEVELS, SCALE_OUT_LEVELS, MAX_POSITION_DURATION, MIN_LIQUIDITY_THRESHOLD,
    INPUT_DIM, SEQUENCE_LENGTH, WINDOW_SIZE, PREDICTION_WINDOW, EVAL_DAYS, VOLATILITY_ADJUSTMENT,
    BATCH_SIZE, LEARNING_RATE, GAMMA, TAU, ENTROPY_COEF, VF_COEF, MAX_GRAD_NORM,
    N_STEPS, N_EPOCHS, N_ENVS, REWARD_SCALE, PROGRESS_INTERVAL
)


# Standard library imports
import json

# ========== Model Definition ==========
class EnhancedTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        # Ensure d_model is divisible by nhead to avoid dimension errors
        if d_model % nhead != 0:
            # Adjust d_model to be divisible by nhead
            d_model = (d_model // nhead) * nhead
            
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="relu"
        )
        self.norm = LayerNorm(d_model)

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.bn1, nn.ReLU(), self.dropout1,
                               self.conv2, self.bn2, nn.ReLU(), self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
        
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
            
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                   stride=1, dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size,
                                   dropout=dropout)]
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleCNN, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels//3, kernel_size=1)
        self.conv3x3 = nn.Conv1d(in_channels, out_channels//3, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv1d(in_channels, out_channels//3, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x1 = self.conv1x1(x)
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        x = torch.cat([x1, x3, x5], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        return x

class EnhancedEnsembleModel(nn.Module):
    """Ensemble model that combines transformer, CNN, and traditional ML models"""
    def __init__(self, params):
        super().__init__()
        self.input_dim = params['input_dim']
        self.embed_dim = params['embed_dim']
        self.device = params.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Log input dimensions for debugging
        logger.info(f"Initializing EnhancedEnsembleModel with input_dim={self.input_dim}, embed_dim={self.embed_dim}")
        
        # Simpler transformer architecture with more attention heads
        transformer_params = params.copy()
        transformer_params['input_dim'] = self.input_dim  # Ensure input_dim is properly set
        self.transformer = SimplifiedTransformer(transformer_params)
        
        # CNN path - ensure it gets the correct input_dim
        cnn_params = params.copy()
        cnn_params['input_dim'] = self.input_dim  # Explicitly set input_dim
        self.cnn = LightweightCNN(cnn_params)
        
        # Feature combination
        self.combiner = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Market regime detection module
        self.regime_detector = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 3 market regimes: trending, ranging, volatile
            nn.Softmax(dim=-1)
        )
        
        # Mixture of experts based on market regime
        self.trend_expert = nn.Linear(self.embed_dim, self.embed_dim)
        self.range_expert = nn.Linear(self.embed_dim, self.embed_dim)
        self.volatile_expert = nn.Linear(self.embed_dim, self.embed_dim)
        
        # Final projection layer
        self.projection = nn.Linear(self.embed_dim, self.embed_dim)
    
    def forward(self, x):
        # Get embeddings from transformer and CNN paths
        transformer_embedding = self.transformer(x)
        cnn_embedding = self.cnn(x)
        
        # Combine the embeddings
        combined = torch.cat([transformer_embedding, cnn_embedding], dim=1)
        features = self.combiner(combined)
        
        # Detect market regime
        regime_weights = self.regime_detector(features)
        
        # Apply mixture of experts
        trend_out = self.trend_expert(features) * regime_weights[:, 0:1]
        range_out = self.range_expert(features) * regime_weights[:, 1:2]
        volatile_out = self.volatile_expert(features) * regime_weights[:, 2:3]
        
        # Combine expert outputs
        expert_out = trend_out + range_out + volatile_out
        
        # Final projection
        output = self.projection(expert_out)
        return output


class SimplifiedTransformer(nn.Module):
    """Simplified transformer architecture with configurable attention heads"""
    def __init__(self, params):
        super().__init__()
        self.input_dim = params['input_dim']
        self.embed_dim = params['embed_dim']
        self.seq_len = params.get('seq_len', 20)
        
        # Log initialization
        logger.info(f"Initializing SimplifiedTransformer with input_dim={self.input_dim}, embed_dim={self.embed_dim}")
        
        # Make number of heads configurable, default to 8
        self.num_heads = params.get('num_heads', 8)
        
        # Validate input dimensions
        if self.input_dim <= 0:
            raise ValueError(f"Invalid input_dim: {self.input_dim}. Must be positive.")
            
        if self.embed_dim <= 0:
            raise ValueError(f"Invalid embed_dim: {self.embed_dim}. Must be positive.")
        
        # Validate that embed_dim is divisible by num_heads
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        
        # Input projection with initialization
        self.input_projection = nn.Linear(self.input_dim, self.embed_dim)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.constant_(self.input_projection.bias, 0.0)
        
        # Position encoding
        self.position_encoding = nn.Parameter(
            torch.zeros(1, self.seq_len, self.embed_dim)
        )
        nn.init.normal_(self.position_encoding, mean=0, std=0.02)
        
        # Transformer with configurable number of heads
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=self.num_heads,
                dim_feedforward=self.embed_dim*2,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ),
            num_layers=2
        )
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        try:
            # Ensure input is a tensor and handle different input formats
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
            else:
                x = x.to(dtype=torch.float32, device=next(self.parameters()).device)
            
            # Handle input shapes: [batch, seq_len, features] or [seq_len, features]
            if len(x.shape) == 2:
                x = x.unsqueeze(0)  # Add batch dimension if missing
            
            batch_size, seq_len, input_dim = x.shape
            
            # Ensure input has the correct number of features
            if input_dim != self.input_dim:
                if input_dim < self.input_dim:
                    # Pad with zeros if input has fewer features
                    padding = torch.zeros(batch_size, seq_len, self.input_dim - input_dim,
                                       device=x.device, dtype=x.dtype)
                    x = torch.cat([x, padding], dim=2)
                else:
                    # Truncate if input has more features
                    x = x[:, :, :self.input_dim]
            
            # Project input to embedding dimension
            x = self.input_projection(x)
            
            # Add positional encoding
            if seq_len > self.seq_len:
                # If sequence is longer than our position encoding, truncate it
                x = x[:, :self.seq_len, :]
                seq_len = self.seq_len
            
            # Ensure position encoding is on the same device as input
            position_encoding = self.position_encoding.to(x.device)
            x = x + position_encoding[:, :seq_len, :]
            
            # Apply transformer
            transformer_out = self.transformer_encoder(x)
            
            # Global pooling to get a single vector per sequence
            # Reshape for pooling: [batch_size, embed_dim, seq_len]
            transformer_out = transformer_out.transpose(1, 2)
            pooled = self.global_avg_pool(transformer_out).squeeze(2)
            
            return pooled
            
        except Exception as e:
            import traceback
            error_msg = f"Error in SimplifiedTransformer forward pass: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            
            # Return zeros tensor with the expected shape on the correct device
            batch_size = x.shape[0] if hasattr(x, 'shape') and len(x.shape) > 0 else 1
            device = next(self.parameters()).device if hasattr(self, 'parameters') else 'cpu'
            return torch.zeros(batch_size, self.embed_dim, device=device, dtype=torch.float32)


class LightweightCNN(nn.Module):
    """Lightweight CNN for feature extraction"""
    def __init__(self, params):
        super().__init__()
        # Make input_dim configurable with a default of 11 for backward compatibility
        self.input_dim = int(params.get('input_dim', 11))  # Configurable input dimension
        self.embed_dim = int(params.get('embed_dim', 256))
        
        # Device initialization
        if 'device' in params:
            device = params['device']
            self.device = torch.device(device) if isinstance(device, str) else device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        logger.info(f"Initialized LightweightCNN with input_dim={self.input_dim}, embed_dim={self.embed_dim}, device={self.device}")
        
        # Validate dimensions
        if self.input_dim <= 0 or self.embed_dim <= 0:
            raise ValueError(f"Invalid dimensions: input_dim={self.input_dim}, embed_dim={self.embed_dim}. All must be positive.")
        
        # Simple CNN backbone
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, self.embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(),
        )
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
          # Move model to correct device
        self.to(self.device)
        
    def forward(self, x):
        try:
            # Get the device from the model parameters
            device = next(self.parameters()).device

            # Convert to tensor if not already
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32, device=device)
            else:
                x = x.to(device=device, dtype=torch.float32)
            
            # Input shape: [batch, seq_len, features] or [seq_len, features]
            if len(x.shape) == 2:
                # Add batch dimension if missing
                x = x.unsqueeze(0)
            elif len(x.shape) == 1:
                # Handle 1D input by adding both batch and sequence dimensions
                x = x.unsqueeze(0).unsqueeze(0)
                
            # Ensure we have the right number of dimensions
            if len(x.shape) != 3:
                raise ValueError(f"Expected input with 2 or 3 dimensions, got {len(x.shape)} dimensions")
                
            batch_size, seq_len, input_dim = x.shape
            
            # Ensure input has the correct number of features
            if input_dim != self.input_dim:
                logger.warning(f"CNN Input dimension mismatch: expected {self.input_dim}, got {input_dim}. Adjusting...")
                # Handle dimension mismatch by padding or truncating
                if input_dim < self.input_dim:
                    # Pad with zeros if input has fewer features
                    padding = torch.zeros(batch_size, seq_len, self.input_dim - input_dim, 
                                       device=device, dtype=torch.float32)
                    x = torch.cat([x, padding], dim=2)
                else:
                    # Truncate if input has more features
                    x = x[:, :, :self.input_dim]
            
            # Transpose to [batch, features, seq_len] for Conv1d
            x = x.transpose(1, 2)
            
            # Apply CNN layers
            x = self.conv_layers(x)
            
            # Global pooling
            x = self.global_avg_pool(x).squeeze(2)
            
            return x
            
        except Exception as e:
            error_msg = f"Error in LightweightCNN forward pass: {str(e)}\nInput shape: {x.shape if hasattr(x, 'shape') else 'N/A'}\n{traceback.format_exc()}"
            logger.error(error_msg)
            
            # Return zeros tensor with the expected shape on the correct device
            batch_size = x.shape[0] if hasattr(x, 'shape') and len(x.shape) > 0 else 1
            device = next(self.parameters()).device
            return torch.zeros(batch_size, self.embed_dim, device=device)


# Original CNNTransformer kept for backward compatibility
class CNNTransformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        # Store dimensions for compatibility with integer types
        self.input_dim = int(params.get('input_dim', 11))
        self.embed_dim = int(params.get('embed_dim', 256))
        self.seq_len = int(params.get('seq_len', 20))

        # Initialize device consistently
        if 'device' in params:
            device = params['device']
            self.device = torch.device(device) if isinstance(device, str) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Log model initialization
        logger.info(f"Initializing CNNTransformer with input_dim={self.input_dim}, embed_dim={self.embed_dim}, seq_len={self.seq_len}, device={self.device}")
        
        # Input validation
        if self.input_dim <= 0 or self.embed_dim <= 0 or self.seq_len <= 0:
            error_msg = f"Invalid dimensions: input_dim={self.input_dim}, embed_dim={self.embed_dim}, seq_len={self.seq_len}. All must be positive."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Create enhanced ensemble model with consistent device
        params['device'] = str(self.device)  # Ensure device is passed as string
        try:
            self.enhanced_model = EnhancedEnsembleModel(params)
            logger.info("Successfully initialized EnhancedEnsembleModel")
        except Exception as e:
            error_msg = f"Failed to initialize EnhancedEnsembleModel: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
              # Move model to device
        self.to(self.device)
        
    def forward(self, x):
        """Forward pass of the CNNTransformer."""
        try:
            # Ensure input is a tensor
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            
            # Move to correct device
            device_str = str(self.device)  # Convert device to string representation
            x = x.to(device=device_str, dtype=torch.float32)
            
            # Handle input shapes: [features], [seq_len, features], or [batch, seq_len, features]
            if len(x.shape) == 1:
                x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, features]
            elif len(x.shape) == 2:
                # Either [seq_len, features] or [batch, features]
                if x.shape[0] == self.input_dim or x.shape[1] == self.input_dim:
                    # [seq_len, features] case
                    x = x.unsqueeze(0) if x.shape[1] == self.input_dim else x.unsqueeze(1)
                else:
                    x = x.unsqueeze(1)  # [batch, 1, features]
            
            # Validate tensor shape
            if len(x.shape) != 3:
                raise ValueError(f"Expected input with 3 dimensions after reshaping, got {len(x.shape)} dimensions")
            
            batch_size, seq_len, input_dim = x.shape
            
            # Handle dimension mismatch by padding or truncating
            if input_dim != self.input_dim:
                if input_dim < self.input_dim:
                    padding = torch.zeros(batch_size, seq_len, self.input_dim - input_dim, 
                                       device=x.device, dtype=x.dtype)
                    x = torch.cat([x, padding], dim=2)
                else:
                    x = x[:, :, :self.input_dim]
            
            # Forward pass through enhanced model
            output = self.enhanced_model(x)
            
            # Handle model output
            if isinstance(output, (tuple, list)):
                output = output[0]
            
            if not isinstance(output, torch.Tensor):
                output = torch.tensor(output, dtype=torch.float32, device=self.device)
            
            # Ensure consistent output shape [batch_size, embed_dim]
            if len(output.shape) > 2:
                output = output.mean(dim=1)
            elif len(output.shape) == 1:
                output = output.unsqueeze(0)
            
            if output.shape[1] > self.embed_dim:
                output = output[:, :self.embed_dim]
            elif output.shape[1] < self.embed_dim:
                padding = torch.zeros(batch_size, self.embed_dim - output.shape[1], 
                                   device=output.device, dtype=output.dtype)
                output = torch.cat([output, padding], dim=1)
            
            return output
            
        except Exception as e:
            logger.error(f"Error in CNNTransformer forward pass: {str(e)}\n{traceback.format_exc()}")
            batch_size = x.shape[0] if hasattr(x, 'shape') and len(x.shape) > 0 else 1
            return torch.zeros(batch_size, self.embed_dim, device=self.device, dtype=torch.float32)
            logger.error(f"Error in CNNTransformer forward pass: {str(e)}\n{traceback.format_exc()}")
            batch_size = x.shape[0] if hasattr(x, 'shape') and len(x.shape) > 0 else 1
            return torch.zeros(batch_size, self.embed_dim, device=self.device)

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        # Calculate attention weights
        att_weights = F.softmax(self.attention(x), dim=1)  # (batch, seq_len, 1)
        # Apply attention weights to input
        context = torch.sum(att_weights * x, dim=1)  # (batch, hidden_dim)
        return context, att_weights

class CurriculumWrapperOld(gym.Wrapper):
    def __init__(self, env, total_timesteps=None):
        super().__init__(env)
        self.env = env
        self.total_timesteps = total_timesteps
        print(f"[DEBUG] CurriculumWrapper total_timesteps: {self.total_timesteps}")
        self.initial_difficulty = 0.1  # Start with low difficulty
        self.difficulty = self.initial_difficulty
        self.initial_reward_factor = 2.5  # Higher initial reward for easier learning
        self.current_step = 0
        self.max_steps = len(self.env.prices) - 1
        self.progress = 0  # Track overall progress through curriculum
        
        # Market regime tracking
        self.current_regime = 'normal'  # Start with normal regime
        self.regime_duration = 200  # Steps to stay in one regime
        self.regime_step = 0
        self.regimes = ['low_volatility', 'normal', 'high_volatility', 'trending', 'ranging']
        self.regime_weights = [0.2, 0.3, 0.2, 0.15, 0.15]  # Probability weights for regimes
        
        # Performance tracking for adaptive difficulty
        self.performance_window = []  # Recent episode returns
        self.window_size = 5  # Number of episodes to average
        self.target_success_rate = 0.6  # Target win rate
        self.success_threshold = 0.02  # Return threshold to count as success
        
        # Initialize environment's reward_factor
        self.env.reward_factor = self.initial_reward_factor
        
        # Create a random number generator for controlled randomness
        self.rng = np.random.RandomState(42)
        
    def update_difficulty(self):
        """Adaptively adjust difficulty based on agent performance"""
        if len(self.performance_window) >= self.window_size:
            # Calculate success rate (positive returns over threshold)
            successes = sum(1 for ret in self.performance_window if ret > self.success_threshold)
            success_rate = successes / len(self.performance_window)
            
            # Adjust difficulty
            if success_rate > self.target_success_rate:
                # Agent is doing well, make it harder
                self.difficulty = min(1.0, self.difficulty + 0.05)
            elif success_rate < self.target_success_rate - 0.2:  # Allow some margin
                # Agent is struggling, make it easier
                self.difficulty = max(self.initial_difficulty, self.difficulty - 0.05)
            
            # Reset window after adjustment
            self.performance_window = self.performance_window[-self.window_size//2:]
    
    def select_market_regime(self):
        """Selects a market regime based on current difficulty and progress"""
        # Regime probabilities change with difficulty
        # As difficulty increases, increase probability of harder regimes
        adjusted_weights = self.regime_weights.copy()
        
        # Increase high volatility and ranging probability with difficulty
        volatility_idx = self.regimes.index('high_volatility')
        ranging_idx = self.regimes.index('ranging')
        adjusted_weights[volatility_idx] *= (1.0 + self.difficulty)
        adjusted_weights[ranging_idx] *= (1.0 + self.difficulty)
        
        # Normalize weights
        adjusted_weights = [w / sum(adjusted_weights) for w in adjusted_weights]
        
        # Select regime
        self.current_regime = self.rng.choice(self.regimes, p=adjusted_weights)
        self.regime_step = 0
        return self.current_regime
    
    def apply_regime_effects(self):
        """Apply effects of current market regime to the environment"""
        # Default parameters
        volatility_factor = 1.0
        transaction_cost = self.env.transaction_cost
        stop_loss = 0.05 - (0.03 * self.progress)  # Base setting
        take_profit = 0.10 - (0.07 * self.progress)  # Base setting
        reward_scale = 1.0
        
        # Modify parameters based on regime
        if self.current_regime == 'low_volatility':
            volatility_factor = 0.5
            transaction_cost *= 0.8  # Lower costs in low volatility
            stop_loss *= 0.8  # Tighter stops in low volatility
            take_profit *= 0.8  # Smaller targets in low volatility
            reward_scale = 0.9  # Slightly lower rewards (harder to make profits)
            
        elif self.current_regime == 'high_volatility':
            volatility_factor = 2.0
            transaction_cost *= 1.2  # Higher costs in high volatility
            stop_loss *= 1.5  # Wider stops needed in high volatility
            take_profit *= 1.3  # Larger targets possible in high volatility
            reward_scale = 1.1  # Slightly higher rewards (more opportunities)
            
        elif self.current_regime == 'trending':
            volatility_factor = 1.2
            reward_scale = 1.2  # Higher rewards for trend following
            
        elif self.current_regime == 'ranging':
            volatility_factor = 0.8
            stop_loss *= 1.2  # Wider stops in ranging markets
            reward_scale = 0.8  # Lower rewards (harder to make profits in ranges)
        
        # Apply effects to environment
        # Scale volatility without actually changing the data
        self.env.volatility_scale = volatility_factor
        self.env.transaction_cost = transaction_cost
        self.env.stop_loss_pct = stop_loss
        self.env.take_profit_pct = take_profit
        self.env.reward_factor = self.initial_reward_factor * reward_scale * (1.5 - 0.5 * self.progress)
    
    def step(self, action):
        # Update progress through curriculum (0 to 1)
        self.progress = min(1.0, self.current_step / (self.max_steps * 0.7))  # Cap at 70% of the way through
        
        # Check if we need to update the market regime
        self.regime_step += 1
        if self.regime_step >= self.regime_duration:
            self.select_market_regime()
        
        # Apply current regime effects
        self.apply_regime_effects()
        
        # Execute step and handle both Gymnasium (5 values) and OpenAI Gym (4 values) formats
        result = self.env.step(action)
        
        if len(result) == 5:
            # Gymnasium format: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = result
        elif len(result) == 4:
            # OpenAI Gym format: (obs, reward, done, info)
            obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            raise ValueError(f"Unexpected number of return values from env.step(): {len(result)}")
        
        # Track episode performance for adaptive difficulty
        if terminated or truncated:
            # Record normalized portfolio return
            if 'final_value' in info:
                normalized_return = (info['final_value'] - START_CAP) / START_CAP
                self.performance_window.append(normalized_return)
                self.update_difficulty()
        
        self.current_step += 1
        return obs, reward, terminated, truncated, info
        
    def reset(self, seed=None, options=None, **kwargs):
        # Reset curriculum parameters
        self.current_step = 0
        self.progress = 0
        self.regime_step = 0
        
        # Select initial market regime
        self.select_market_regime()
        
        # Apply initial regime effects
        self.apply_regime_effects()
        
        # Reset environment
        result = self.env.reset(seed=seed, options=options, **kwargs)
        
        # Initialize environment specific curriculum parameters
        if hasattr(self.env, 'volatility_scale'):
            self.env.volatility_scale = 1.0
        
        return result

class HybridModel(nn.Module):
    def __init__(self, params):
        super(HybridModel, self).__init__()
        # Extract parameters
        input_dim = params['input_dim']
        hidden_dim = params.get('hidden_dim', 128)
        embed_dim = params['embed_dim']
        dropout = params.get('dropout_rate', 0.3)
        num_layers = params.get('num_layers', 2)
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 0 else 0,
            bidirectional=True
        )
        
        lstm_out_dim = hidden_dim * 2  # bidirectional
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_out_dim,
            num_heads=params.get('num_heads', 8),
            dropout=dropout
        )
        
        # Shared features layer
        self.shared_features = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Head 1: Profit Classification (binary)
        self.profit_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )
        
        # Head 2: Stop Loss Prediction
        self.stop_loss_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Normalize to [0,1] range
        )
        
        # Head 3: Take Profit Prediction
        self.take_profit_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Normalize to [0,1] range
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
                    
    def forward(self, x):
        # Extract features
        batch_size = x.size(0)
        features = self.feature_extractor(x)
        
        # Process sequence
        lstm_out, _ = self.lstm(features)
        
        # Apply attention
        attn_out, _ = self.attention(
            lstm_out.transpose(0, 1),  # Convert to seq_len, batch, features
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1)
        )
        attn_out = attn_out.transpose(0, 1)  # Back to batch, seq_len, features
        
        # Get final timestep features
        final_features = attn_out[:, -1, :]  # Use last timestep
        
        # Get shared features
        shared = self.shared_features(final_features)
        
        # Head outputs
        profit_logits = self.profit_head(shared)  # Classification logits
        stop_loss = self.stop_loss_head(shared)  # Stop loss percentage
        take_profit = self.take_profit_head(shared)  # Take profit percentage
        
        # Scale the regression outputs to reasonable ranges
        stop_loss = stop_loss * 0.1  # Max 10% stop loss
        take_profit = take_profit * 0.2  # Max 20% take profit
        
        return profit_logits, stop_loss, take_profit
        
    def calculate_loss(self, profit_logits, stop_loss, take_profit, 
                      profit_targets, sl_targets, tp_targets, 
                      profit_criterion, sl_criterion, tp_criterion):
        """Calculate combined loss from all three heads"""
        # Classification loss for profit prediction
        profit_loss = profit_criterion(profit_logits, profit_targets)
        
        # Regression losses for stop loss and take profit
        sl_loss = sl_criterion(stop_loss.squeeze(), sl_targets)
        tp_loss = tp_criterion(take_profit.squeeze(), tp_targets)
        
        # Combine losses with weighting
        total_loss = (0.5 * profit_loss +  # Higher weight for main task
                     0.25 * sl_loss +      # Equal weights for risk management
                     0.25 * tp_loss)       # tasks
                     
        return total_loss, {
            'profit_loss': profit_loss.item(),
            'sl_loss': sl_loss.item(),
            'tp_loss': tp_loss.item()
        }

class SB3FeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor for Stable-Baselines3 with CNN-Transformer architecture."""
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, **kwargs):
        """Initialize the feature extractor.
        
        Args:
            observation_space: The observation space
            features_dim: Number of features to extract
            **kwargs: Additional arguments
        """
        self._features_dim = features_dim
        super().__init__(observation_space, features_dim)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get input shape from observation space
        if isinstance(observation_space, spaces.Box) and observation_space.shape is not None:
            # Convert shape to list then numpy array for safer indexing and type consistency
            _shape = list(observation_space.shape)
            self.input_shape = np.array(_shape)
            if self.input_shape.size == 0:
                logger.warning("Empty observation space shape, using default [features_dim]")
                self.input_shape = np.array([features_dim])
        else:
            logger.warning("Observation space is not Box type or shape is None, using default [features_dim]")
            self.input_shape = np.array([features_dim])
        
        # Initialize the CNN-Transformer model with proper input dimensions
        input_dim = features_dim
        seq_len = 1
        try:
            # Handle different input shapes safely
            if self.input_shape.ndim > 1 : # typically (seq_len, num_features) or (batch, seq_len, num_features)
                input_dim = int(self.input_shape[-1]) 
                seq_len = int(self.input_shape[-2]) if self.input_shape.ndim > 1 else 1
            elif self.input_shape.ndim == 1 and self.input_shape.size > 0 : # typically (num_features) for flat obs
                input_dim = int(self.input_shape[0])
                seq_len = 1 # For a flat observation, sequence length is 1
            else: # Fallback for empty or unexpected shapes
                logger.warning(f"Unexpected input_shape: {self.input_shape}. Using default input_dim={features_dim}, seq_len=1")
                input_dim = features_dim
                seq_len = 1

        except (IndexError, TypeError, AttributeError) as e: # Added TypeError
            logger.warning(f"Error processing input shape {self.input_shape}, using default dimensions input_dim={features_dim}, seq_len=1: {str(e)}")
            input_dim = features_dim
            seq_len = 1
        
        # Set up model parameters
        self.params = {
            'input_dim': input_dim,
            'embed_dim': features_dim, # features_dim is the output dim of the feature extractor
            'seq_len': seq_len,
            'num_heads': 8,  # Default number of attention heads
            'device': self.device
        }
        
        self.model = CNNTransformer(self.params)
        self.model.to(self.device)
    
    @property
    def features_dim(self) -> int:
        return self._features_dim
    
    def _ensure_float_tensor(self, x: Any) -> torch.Tensor: # Changed type hint for x
        """Ensure input is a float32 tensor on the correct device."""
        if not isinstance(x, torch.Tensor):
            # Attempt to convert numpy array or list to tensor
            try:
                x = torch.tensor(x, dtype=torch.float32)
            except Exception as e:
                logger.error(f"Failed to convert input to tensor: {x}, error: {e}")
                # Fallback: create a zero tensor with expected feature dimension if conversion fails
                # This part is tricky as we don't know batch size here.
                # Consider raising an error or handling it based on where this util is called.
                # For SB3, observations usually come in batches.
                # If this is called from forward, batch size is known.
                # For now, let's assume it's a single observation if not a tensor.
                return torch.zeros((1, self.params.get('input_dim', self._features_dim)), dtype=torch.float32, device=self.device)

        return x.to(device=self.device, dtype=torch.float32)

    def _reshape_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Reshape observations to the expected (batch_size, seq_len, features)."""
        # Expected input for CNNTransformer: (batch_size, seq_len, features)
        # self.params['seq_len'] and self.params['input_dim'] should be correct
        
        # if observations.shape == torch.Size([1]): # Handle scalar tensor
        #     observations = observations.reshape(1,1,1)

        if observations.ndim == 1: # (features,)
            # Assume batch_size=1, seq_len=1
            observations = observations.unsqueeze(0).unsqueeze(0) 
        elif observations.ndim == 2: # (batch_size, features) or (seq_len, features)
            # If it's (batch_size, features), assume seq_len=1
            # If it's (seq_len, features) from internal model, it might need batch_size=1
            # For SB3, it's usually (batch_size, features) if flat, or (batch_size, seq_len, features) if image-like
            # Let's assume (batch_size, features) and add seq_len dimension
             observations = observations.unsqueeze(1) # -> (batch_size, 1, features)
        # If ndim == 3, assume it's already (batch_size, seq_len, features)
        
        # Ensure it matches the model's expected seq_len and input_dim if possible
        # This can be complex if seq_len varies. For now, rely on SB3 providing consistent shapes.
        return observations    
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor: # Keep torch.Tensor hint for SB3
        """Forward pass of the feature extractor."""
        batch_size = 1 # Default batch size

        try:
            # Ensure observations is a tensor and on the correct device
            if not isinstance(observations, torch.Tensor):
                # This handles numpy arrays typically passed by SB3
                observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
            else:
                observations = observations.to(device=self.device, dtype=torch.float32)

            # Determine batch_size after tensor conversion
            if observations.ndim > 0:
                batch_size = observations.shape[0]
            else: # Handle scalar tensor if it somehow occurs
                observations = observations.reshape(1,1,1) # Make it (1,1,1)
                batch_size = 1
            
            # Reshape based on number of dimensions
            # SB3 typically passes (batch_size, *observation_shape)
            # For flat features: (batch_size, feature_dim)
            # For sequence/image: (batch_size, seq_len, feature_dim) or (batch_size, H, W, C)

            # Ensure self.observation_space.shape is treated as a tuple before converting to list
            space_shape_tuple = tuple(self.observation_space.shape)
            obs_shape_list = list(space_shape_tuple) # Use the original observation_space shape

            if len(obs_shape_list) == 1: # Flat observation (num_features,)
                # Expected by model: (batch_size, seq_len, num_features)
                # Here, seq_len is 1.
                if observations.ndim == 1: # single observation (features,) - should be rare from SB3
                    observations = observations.unsqueeze(0).unsqueeze(1) # (1, 1, features)
                elif observations.ndim == 2: # (batch_size, features)
                    observations = observations.unsqueeze(1) # (batch_size, 1, features)
            elif len(obs_shape_list) > 1: # Already has sequence or image-like structure
                if observations.ndim == len(obs_shape_list): # single observation (seq_len, features)
                    observations = observations.unsqueeze(0) # (1, seq_len, features)
                # if observations.ndim == len(obs_shape_list) + 1, it's (batch, seq_len, features), which is fine.
            
            # Final check on dimensions before passing to the model
            # Model expects (batch_size, seq_len, input_dim)
            # self.params['seq_len'] and self.params['input_dim'] are from __init__
            # This part can be tricky if the actual observation shape dynamically changes
            # For now, let's ensure the last dimension matches input_dim and middle is seq_len

            if observations.ndim == 3:
                if observations.shape[2] != self.params['input_dim']:
                    logger.warning(f"Feature dimension mismatch: obs.shape[2]={observations.shape[2]}, expected={self.params['input_dim']}. Reshaping might be needed or model params incorrect.")
                # if observations.shape[1] != self.params['seq_len']:
                #    logger.warning(f"Sequence length mismatch: obs.shape[1]={observations.shape[1]}, expected={self.params['seq_len']}.")
            else:
                logger.warning(f"Observations not in 3D (batch, seq, feature) format after reshaping: {observations.shape}")


            # Forward pass through the model
            features = self.model(observations) # Model expects (batch_size, seq_len, input_dim)
            
            # The output of CNNTransformer is typically (batch_size, embed_dim) or (batch_size, seq_len, embed_dim)
            # SB3 expects (batch_size, features_dim)
            if features.ndim == 3: # (batch_size, seq_len, embed_dim)
                # Take the features of the last sequence step or average, depending on model
                # Assuming CNNTransformer's forward already handles this and returns (batch_size, features_dim)
                # If not, this might need: features = features[:, -1, :]
                pass # Let's assume model's forward returns correct shape for now.

            if features.shape[-1] != self._features_dim:
                 logger.error(f"Output feature dimension mismatch! Expected {self._features_dim}, got {features.shape[-1]}")
                 # Attempt to fix or create zeros
                 # This is a critical error, indicates model definition or feature_dim param is wrong
                 # Forcing it to self._features_dim might hide issues.
                 # Create zeros as a last resort to avoid crashing.
                 features = torch.zeros((batch_size, self._features_dim), dtype=torch.float32, device=self.device)


            # Ensure output has correct dimensions (batch_size, self._features_dim)
            if features.ndim == 1: # If model somehow returns a 1D tensor
                features = features.unsqueeze(0) # Assume batch_size = 1
            
            # If features are (batch_size, seq_len, features_dim), SB3 might want (batch_size, features_dim)
            # This depends on how the CNNTransformer is implemented.
            # For now, let's assume CNNTransformer outputs (batch_size, self._features_dim)
            # or (batch_size, seq_len, self._features_dim) and we need to pool/flatten if the latter.
            # The CNNTransformer currently seems to output (batch_size, seq_len, embed_dim).
            # We need to ensure it's (batch_size, self._features_dim).
            # Let's take the output of the last time step if it's sequential.

            if features.ndim == 3 and features.shape[1] > 1: # (batch, seq, embed_dim)
                features = features[:, -1, :] # Take last time step's features

            # Final check: features should be (batch_size, self._features_dim)
            if features.ndim == 2 and features.shape[0] == batch_size and features.shape[1] == self._features_dim:
                return features
            elif features.ndim == 2 and features.shape[1] != self._features_dim :
                 logger.error(f"CRITICAL: Final features dim mismatch. Expected {self._features_dim}, got {features.shape[1]}. Returning Zeros.")
                 return torch.zeros((batch_size, self._features_dim),dtype=torch.float32,device=self.device)
            else: # Fallback if something is still wrong
                logger.warning(f"Unexpected final feature shape: {features.shape}. Reshaping to ({batch_size}, {self._features_dim}) with zeros.")
                return torch.zeros((batch_size, self._features_dim), dtype=torch.float32, device=self.device)
            
        except Exception as e:
            logger.error(f"Error in SB3FeatureExtractor.forward: {str(e)}", exc_info=True) # Add exc_info for traceback
            
            # Return zeros with correct shape on error
            # batch_size might not be correctly determined if error is early
            # Try to get it from observations if it's a tensor
            current_batch_size = 1
            if isinstance(observations, torch.Tensor) and observations.ndim > 0:
                current_batch_size = observations.shape[0]
            
            return torch.zeros(
                (current_batch_size, self._features_dim),
                dtype=torch.float32,
                device=self.device
            )
            
# ========== Multi-Agent RL System ==========
class MultiAgentEnsemble:
    """Ensemble of RL agents with different risk profiles for robust trading decisions"""
    def __init__(self, env, base_model=None, n_agents=3):
        self.env = env
        self.n_agents = int(n_agents)  # Ensure integer
        self.agents = []
        self.weights = np.ones(self.n_agents, dtype=np.float32) / self.n_agents  # Use float32
        self.agent_names = ['conservative', 'balanced', 'aggressive']
        self.performance_history = {name: [] for name in self.agent_names}
        self.base_model = base_model
        self.initialize_agents()

    def initialize_agents(self):
        """Initialize agents with different risk profiles"""
        # Conservative agent: Lower risk, more stable returns
        conservative_params = {
            'ent_coef': np.float32(0.05),
            'learning_rate': np.float32(0.0002),
            'gamma': np.float32(0.98),
            'risk_factor': np.float32(0.5),
            'transaction_cost': np.float32(0.002),
            'stop_loss_pct': np.float32(0.015),
            'take_profit_pct': np.float32(0.03),
            'risk_per_trade': np.float32(0.01)
        }
        
        # Balanced agent: Moderate risk/reward
        balanced_params = {
            'ent_coef': np.float32(0.03),
            'learning_rate': np.float32(0.0003),
            'gamma': np.float32(0.95),
            'risk_factor': np.float32(1.0),
            'transaction_cost': np.float32(0.001),
            'stop_loss_pct': np.float32(0.02),
            'take_profit_pct': np.float32(0.04),
            'risk_per_trade': np.float32(0.02)
        }
        
        # Aggressive agent: Higher risk for higher returns
        aggressive_params = {
            'ent_coef': np.float32(0.01),
            'learning_rate': np.float32(0.0005),
            'gamma': np.float32(0.92),
            'risk_factor': np.float32(2.0),
            'transaction_cost': np.float32(0.0005),
            'stop_loss_pct': np.float32(0.03),
            'take_profit_pct': np.float32(0.06),
            'risk_per_trade': np.float32(0.04)
        }
        
        # Create each agent
        if self.base_model:
            # Clone from base model if provided
            self.agents = []
            for i, (name, params) in enumerate(zip(self.agent_names, 
                                               [conservative_params, balanced_params, aggressive_params])):
                # Clone and modify base model
                agent = self.base_model.learn(0)  # Clone without training
                # Apply agent-specific parameters as float32
                for param_name, param_value in params.items():
                    if hasattr(agent, param_name):
                        setattr(agent, param_name, np.float32(param_value))
                self.agents.append(agent)
        else:
            # Create new models with properly typed parameters
            for i, (name, params) in enumerate(zip(self.agent_names, 
                                               [conservative_params, balanced_params, aggressive_params])):
                # Use appropriate policy based on environment
                policy = "MlpPolicy"
                model_class = PPO  # Default to PPO
                
                # Create agent with profile-specific parameters
                agent_params = {
                    'policy': policy,
                    'env': self.env,
                    'verbose': 0,
                }
                # Add parameters that PPO accepts
                ppo_params = ['ent_coef', 'learning_rate', 'gamma']
                agent_params.update({k: np.float32(params[k]) for k in ppo_params if k in params})
                
                # Create agent ensuring float32 parameters
                agent = model_class(**agent_params)
                self.agents.append(agent)
    
    def predict(self, observation, state=None, deterministic=False):
        """Generate ensemble prediction by weighted voting of all agents"""
        # Get predictions from all agents
        actions = []
        values = []
        
        for agent in self.agents:
            action, value = agent.predict(observation, state, deterministic)
            actions.append(action)
            values.append(value)
        
        # Two ensemble strategies:
        # 1. Weighted voting for discrete actions
        if isinstance(actions[0], (int, np.integer)) or (isinstance(actions[0], np.ndarray) and actions[0].size == 1):
            # Convert to simple integers if they're single-element arrays
            actions = [a[0] if isinstance(a, np.ndarray) and a.size == 1 else a for a in actions]
            
            # Count votes for each action
            action_votes = {}
            for i, action in enumerate(actions):
                if action not in action_votes:
                    action_votes[action] = 0
                action_votes[action] += self.weights[i]
            
            # Find action with most votes
            ensemble_action = max(action_votes.items(), key=lambda x: x[1])[0]
            
        # 2. Weighted average for continuous actions
        else:
            # Weighted average of continuous actions
            ensemble_action = np.average(np.array(actions), axis=0, weights=self.weights)
        
        # Average value estimate
        ensemble_value = np.average(np.array(values), axis=0, weights=self.weights)
        
        return ensemble_action, ensemble_value
    
    def learn(self, total_timesteps, callback=None, log_interval=100):
        """Train all agents simultaneously and update their weights based on performance"""
        # Train each agent
        for i, agent in enumerate(self.agents):
            logging.info(f"Training {self.agent_names[i]} agent for {total_timesteps} steps")
            agent.learn(total_timesteps=total_timesteps, callback=callback)
        
        # Evaluate and update weights
        self.update_weights()
        
        return self
    
    def update_weights(self):
        """Update agent weights based on their performance"""
        # Evaluate each agent
        performance_scores = []
        
        for i, agent in enumerate(self.agents):
            # Create evaluation environment
            eval_env = self.env
            
            # Run evaluation episodes
            total_reward = np.float32(0.0)
            episode_rewards = []
            
            obs = eval_env.reset()[0]
            done = False
            
            for _ in range(100):  # Evaluate for 100 steps
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, done, _, _ = eval_env.step(action)
                total_reward += np.float32(reward)
                episode_rewards.append(np.float32(reward))
                
                if done:
                    obs = eval_env.reset()[0]
            
            # Calculate metrics using float32
            if episode_rewards:
                mean_reward = total_reward / np.float32(len(episode_rewards))
                std_reward = np.float32(np.std(episode_rewards)) if len(episode_rewards) > 1 else np.float32(1.0)
                sharpe_ratio = mean_reward / (std_reward + np.float32(1e-9))
                win_rate = np.float32(sum(1 for r in episode_rewards if r > 0)) / np.float32(len(episode_rewards))
                
                # Combined performance score with float32
                score = np.float32(mean_reward * 0.5 + sharpe_ratio * 0.3 + win_rate * 0.2)
            else:
                score = np.float32(0.0)
                
            performance_scores.append(score)
            
            # Store performance history
            self.performance_history[self.agent_names[i]].append(float(score))  # Convert to Python float for JSON serialization
        
        # Update weights using softmax of performance scores
        if all(np.isfinite(performance_scores)):
            performance_scores = np.array(performance_scores, dtype=np.float32)
            # Softmax normalization of scores with float32
            max_score = np.float32(np.max(performance_scores))
            exp_scores = np.exp(performance_scores - max_score)  # For numerical stability
            self.weights = exp_scores.astype(np.float32) / np.float32(np.sum(exp_scores))
        
        logging.info(f"Updated agent weights: {list(zip(self.agent_names, self.weights.tolist()))}")
    
    def save(self, path):
        """Save all agents and ensemble parameters"""
        os.makedirs(path, exist_ok=True)
        
        # Save each agent
        for i, agent in enumerate(self.agents):
            agent_path = os.path.join(path, f"{self.agent_names[i]}.zip")
            agent.save(agent_path)
        
        # Save ensemble parameters
        ensemble_params = {
            'weights': self.weights.tolist(),
            'performance_history': self.performance_history
        }
        
        with open(os.path.join(path, 'ensemble_params.json'), 'w') as f:
            json.dump(ensemble_params, f)
    
    def load(self, path, env=None):
        """Load all agents and ensemble parameters"""
        if env is not None:
            self.env = env
        
        # Load each agent
        self.agents = []
        for i, name in enumerate(self.agent_names):
            agent_path = os.path.join(path, f"{name}.zip")
            if os.path.exists(agent_path):
                self.agents.append(PPO.load(agent_path, env=self.env))
        
        # Load ensemble parameters
        params_path = os.path.join(path, 'ensemble_params.json')
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                ensemble_params = json.load(f)
                self.weights = np.array(ensemble_params['weights'])
                self.performance_history = ensemble_params['performance_history']
