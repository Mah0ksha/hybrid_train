# Standard library imports
import os
import time
import logging
import traceback
from collections import deque
from typing import Dict, List, Optional, Tuple, Any, Union

# Third-party imports
import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.callbacks import BaseCallback

# Import local modules
from logger_setup import logger
from trading_metrics import BaseTradingMetrics, EnhancedTradingMetrics
from config_vals import START_CAP

class UtilsTradingMetrics(BaseTradingMetrics):
    """Local trading metrics utility class."""
    def __init__(self, risk_manager=None):
        super().__init__()
        self.risk_manager = risk_manager
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.peak_value = float(START_CAP)
        self.max_drawdown = 0.0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        self.portfolio_value = float(START_CAP)
        self.trade_history = []
    
    def update(self, trade_info, current_value, position=None):
        """Update metrics with new trade information.
        
        Args:
            trade_info: Trade information (dict or float)
            current_value: Current portfolio value
            position: Current position size (optional)
        """
        try:
            self.portfolio_value = float(current_value)
            
        if isinstance(trade_info, dict):
                profit = trade_info.get('profit', 0.0)
                position = trade_info.get('position', position)
            else:
                profit = float(trade_info)
                
            if profit > 0:
                self.winning_trades += 1
                self.total_profit += profit
                self.consecutive_losses = 0
            elif profit < 0:
                self.losing_trades += 1
                self.total_loss += abs(profit)
                self.consecutive_losses += 1
                self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
                
            self.total_trades += 1
            self.peak_value = max(self.peak_value, self.portfolio_value)
            current_drawdown = 1.0 - (self.portfolio_value / self.peak_value) if self.peak_value > 0 else 0.0
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Record trade
            self.trade_history.append({
                'profit': profit,
                'portfolio_value': self.portfolio_value,
                'drawdown': current_drawdown,
                'position': position,
                'timestamp': time.time()
            })
            
        except Exception as e:
            logger.error(f"Error updating utils trading metrics: {str(e)}")
            logger.error(traceback.format_exc())
    
    def calculate_metrics(self):
        """Calculate current trading metrics."""
        try:
            metrics = {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': self.winning_trades / max(1, self.total_trades),
                'total_profit': self.total_profit,
                'total_loss': self.total_loss,
                'net_profit': self.total_profit - abs(self.total_loss),
                'max_drawdown': self.max_drawdown,
                'consecutive_losses': self.consecutive_losses,
                'max_consecutive_losses': self.max_consecutive_losses,
                'current_value': self.portfolio_value,
                'peak_value': self.peak_value
            }
            return metrics
        except Exception as e:
            logger.error(f"Error calculating utils trading metrics: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
            
    def reset(self):
        """Reset all metrics to initial values."""
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.peak_value = float(START_CAP)
        self.max_drawdown = 0.0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        self.portfolio_value = float(START_CAP)
        self.trade_history = []

def evaluate_model(model, env, n_eval_episodes=5):
    """
    Evaluate a trained model with enhanced trading metrics.
    
    Args:
        model: The trained RL model
        env: The evaluation environment
        n_eval_episodes: Number of episodes to evaluate
    
    Returns:
        dict: Evaluation statistics including trading metrics
    """
    episode_rewards = []
    episode_lengths = []
    trading_metrics = None
    trade_data = []
    
    # Check if the environment has a risk_manager
    if hasattr(env, 'env') and hasattr(env.env, 'risk_manager'):
        risk_manager = env.env.risk_manager
    elif hasattr(env, 'risk_manager'):
        risk_manager = env.risk_manager
    else:
        risk_manager = None
        logger.warning("No risk manager found in environment for metrics collection")
    
    # Create metrics collector
    metrics_collector = EnhancedTradingMetrics(risk_manager=risk_manager)
    
    try:
        for episode in range(n_eval_episodes):
            if hasattr(env, 'reset'):
                obs_result = env.reset()
                # Handle different gym versions
                if isinstance(obs_result, tuple):
                    obs = obs_result[0]
                else:
                    obs = obs_result
            else:
                logger.warning("Environment has no reset method")
                break
                
            done = False
            episode_reward = 0
            episode_length = 0
            entry_price = None
            entry_time = None
            position_type = None
            current_position = 0
            
            while not done:
                # Get action from model
                try:
                    action, _ = model.predict(obs, deterministic=True)
                except Exception as e:
                    logger.error(f"Error predicting action: {str(e)}")
                    traceback.print_exc()
                    break
                
                # Take action in environment
                try:
                    step_result = env.step(action)
                except Exception as e:
                    logger.error(f"Error stepping environment: {str(e)}")
                    traceback.print_exc()
                    break
                
                # Handle different gym versions
                if len(step_result) == 5:  # New gym
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:  # Old gym
                    obs, reward, done, info = step_result
                
                # Track portfolio value and position
                if hasattr(env, 'env') and hasattr(env.env, 'portfolio_value'):
                    if callable(env.env.portfolio_value):
                        portfolio_value = env.env.portfolio_value()
                    else:
                        portfolio_value = env.env.portfolio_value
                    
                    if hasattr(env.env, 'position'):
                        current_position = env.env.position
                    
                    metrics_collector.update(info.get('trade', 0.0), portfolio_value, current_position)
                elif hasattr(env, 'portfolio_value'):
                    if callable(env.portfolio_value):
                        portfolio_value = env.portfolio_value()
                    else:
                        portfolio_value = env.portfolio_value
                    
                    if hasattr(env, 'position'):
                        current_position = env.position
                    
                    metrics_collector.update(info.get('trade', 0.0), portfolio_value, current_position)
                
                # Track trades
                if info and isinstance(info, dict) and 'trade' in info:
                    trade_info = info['trade']
                    if trade_info:
                        metrics_collector.update(trade_info, metrics_collector.current_value, current_position)
                        trade_data.append(trade_info)
                
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
    
        # Calculate standard statistics
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0
        std_reward = np.std(episode_rewards) if episode_rewards else 0
        mean_length = np.mean(episode_lengths) if episode_lengths else 0
        
        # Calculate trading metrics
        trading_metrics = metrics_collector.calculate_metrics() if hasattr(metrics_collector, 'calculate_metrics') else {}
        
        # Return combined results
        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_length': mean_length,
            'n_episodes': n_eval_episodes,
            'trading_metrics': trading_metrics,
            'trade_data': trade_data
        }
    except Exception as e:
        logger.error(f"Error in evaluate_model: {str(e)}")
        traceback.print_exc()
        return {
            'mean_reward': 0,
            'std_reward': 0,
            'mean_length': 0,
            'n_episodes': n_eval_episodes,
            'trading_metrics': {},
            'error': str(e)
        }


class ProgressCallback(BaseCallback):
    def __init__(self, interval=1.0):
        super().__init__()
        self.interval = interval
        self.last = ""
        self.last_progress_log = 0
        self.progress_log_interval = interval
        
    def _on_step(self) -> bool:
        if time.time() - self.last_progress_log >= self.progress_log_interval:
            steps = self.num_timesteps
            self.last_progress_log = time.time()
            print(f"RL timesteps: {steps}", end="\r")
        return True


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
        
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr