import logging
import time
from datetime import datetime
from typing import Dict, Optional, Any, List, Deque
import pandas as pd
import numpy as np
import os
import json
import torch
from stable_baselines3.common.callbacks import BaseCallback
from performance_monitor import PerformanceMonitor
from logger_setup import setup_logger
from tqdm import tqdm
from collections import OrderedDict, deque

logger = setup_logger()

class EnhancedProgressMonitor(BaseCallback):
    """
    Enhanced monitor for detailed training progress tracking, trading performance monitoring, and NaN detection
    with improved stall detection capabilities.
    
    Args:
        log_freq (int): Frequency of logging progress (default: 100)
        verbose (int): Verbosity level (0: no output, 1: info, 2: debug)
    """
    def __init__(self, log_freq: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.start_time: Optional[float] = None
        self.last_log_time: Optional[float] = None
        self.episode_rewards_window: Deque[float] = deque(maxlen=100)
        self.step_times: Deque[float] = deque(maxlen=log_freq)
        self.metrics_history: List[Dict[str, Any]] = []
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.nan_counts: Dict[str, int] = {'training': 0, 'validation': 0, 'training_actions': 0, 'training_info': 0, 'training_episode_metrics': 0}
        
        self.min_step_rate = 0.1
        self.stall_check_window = 1000
        self.max_training_duration = 7200
        self.consecutive_slow_steps = 0
        self.max_consecutive_slow = 5
        
        self.pbar: Optional[tqdm] = None

    def _init_callback(self) -> None:
        """Initialize timing variables and performance monitor"""
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        model_log_dir = getattr(self.model, 'tensorboard_log', None) if self.model else None
        pm_save_dir_base = model_log_dir if model_log_dir else os.path.join('logs', 'pm_enhanced_monitor')
        pm_save_dir = os.path.join(pm_save_dir_base, 'performance_metrics')
        os.makedirs(pm_save_dir, exist_ok=True)
        
        if self.performance_monitor is None:
            self.performance_monitor = PerformanceMonitor(save_dir=pm_save_dir)

    def check_nan_values(self, data_dict: Dict[str, Any], stage_key_prefix: str = 'training') -> bool:
        """Centralized NaN check for tensors, arrays and values in a dictionary.
        
        Args:
            data_dict: Dictionary of values to check.
            stage_key_prefix: Prefix for the key in self.nan_counts (e.g., 'training_actions').
            
        Returns:
            bool: True if NaN values were found, False otherwise.
        """
        overall_has_nans = False

        def _is_nan(val: Any) -> bool:
            if isinstance(val, torch.Tensor):
                return bool(torch.isnan(val).any().item())
            elif isinstance(val, np.ndarray):
                return bool(np.isnan(val).any())
            elif isinstance(val, (float, int)):
                return bool(np.isnan(val))
            return False

        for key, value in data_dict.items():
            if isinstance(value, dict):
                if self.check_nan_values(value, f"{stage_key_prefix}_{key}"):
                    overall_has_nans = True
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if _is_nan(item):
                        full_key = f"{stage_key_prefix}_{key}[{i}]"
                        self.nan_counts[stage_key_prefix] = self.nan_counts.get(stage_key_prefix, 0) + 1
                        logger.warning(f"NaN detected in {full_key}")
                        overall_has_nans = True
            elif _is_nan(value):
                full_key = f"{stage_key_prefix}_{key}"
                self.nan_counts[stage_key_prefix] = self.nan_counts.get(stage_key_prefix, 0) + 1
                logger.warning(f"NaN detected in {full_key}")
                overall_has_nans = True
                
        return overall_has_nans

    def _on_training_start(self) -> None:
        logger.info("DEBUG: EnhancedProgressMonitor._on_training_start called")
        if self.verbose > 0 and self.num_timesteps is not None:
            self.pbar = tqdm(total=self.num_timesteps, desc="Training Progress")
        elif self.verbose > 0:
            logger.warning("EnhancedProgressMonitor: self.num_timesteps not available, cannot init tqdm progress bar with total.")
            self.pbar = tqdm(desc="Training Progress (total unknown)")

    def _on_step(self) -> bool:
        current_time = time.time()
        if self.last_log_time is not None:
            self.step_times.append(current_time - self.last_log_time)
        self.last_log_time = current_time

        if self.pbar is not None:
            self.pbar.update(1)
        
        if 'actions' in self.locals:
            if self.check_nan_values({'actions': self.locals['actions']}, 'training_actions'):
                 logger.error("NaN detected in model actions. Training might be unstable.")

        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info and info['episode'] is not None:
                    self.episode_rewards_window.append(info['episode']['r'])
                    if self.performance_monitor is not None:
                        metrics_for_pm = {
                            'reward': info['episode']['r'],
                            'length': info['episode']['l'],
                            'profit': info.get('total_profit', 0.0),
                            'trades': info.get('metrics_total_trades', 0),
                            'win_rate': info.get('win_rate', 0.0),
                            'nan_count_train': self.nan_counts.get('training', 0),
                            'nan_count_val': self.nan_counts.get('validation', 0)
                        }
                        if not self.check_nan_values(metrics_for_pm, 'training_episode_metrics'):
                            self.performance_monitor.update_training_metrics(
                                self.n_calls, metrics_for_pm, None
                            )
                info_copy = {k: v for k, v in info.items() if k != 'episode'}
                if self.check_nan_values(info_copy, 'training_info'):
                    logger.warning("NaN detected in environment info dictionary.")

        if self.n_calls % self.log_freq == 0:
            if self.pbar is not None and len(self.episode_rewards_window) > 0:
                mean_reward = np.mean(list(self.episode_rewards_window))
                self.pbar.set_postfix(ordered_dict=OrderedDict([('mean_reward', f'{mean_reward:.2f}')]))
            
            first_info = self.locals['infos'][0] if len(self.locals.get('infos', [])) > 0 else {}
            self._log_progress(current_time, first_info)
            if self.verbose > 0 : logger.info(f"EnhancedProgressMonitor._on_step - call {self.n_calls} logged.")

        return True

    def _log_progress(self, current_time: float, info: Dict[str, Any]) -> None:
        elapsed = current_time - self.start_time if self.start_time is not None else 0.0
        fps = self.n_calls / elapsed if elapsed > 0 else 0
        
        recent_steps_len = len(self.step_times)
        recent_time_sum = sum(self.step_times)
        step_rate = recent_steps_len / recent_time_sum if recent_time_sum > 0 else 0
        
        if step_rate < self.min_step_rate and recent_steps_len >= self.log_freq :
            self.consecutive_slow_steps += 1
            logger.warning(
                f"Potential training stall detected: Step rate {step_rate:.2f} steps/sec "
                f"(below minimum {self.min_step_rate})"
            )
            if self.consecutive_slow_steps >= self.max_consecutive_slow:
                logger.error(
                    f"Critical stall detected: {self.consecutive_slow_steps} consecutive "
                    f"slow windows. Consider stopping training."
                )
            if 'error' in info:
                logger.error(f"Error context from env info: {info.get('error', 'N/A')}")
        else:
            self.consecutive_slow_steps = 0
        
        if elapsed > self.max_training_duration:
            logger.warning(
                f"Training duration ({elapsed/3600:.1f} hours) exceeds maximum "
                f"({self.max_training_duration/3600:.1f} hours) - potential stall"
            )
        
        reward_stats = {}
        if self.episode_rewards_window:
            rewards_arr = np.array(list(self.episode_rewards_window))
            reward_stats = {
                'mean_reward': np.mean(rewards_arr),
                'median_reward': np.median(rewards_arr),
                'min_reward': np.min(rewards_arr),
                'max_reward': np.max(rewards_arr),
                'reward_std': np.std(rewards_arr),
                'reward_trend': np.mean(np.diff(rewards_arr)) if len(rewards_arr) > 1 else 0.0
            }
            
        metrics_log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': self.n_calls,
            'total_time_sec': elapsed,
            'fps': fps,
            'step_rate': step_rate,
            'nan_counts': self.nan_counts.copy(),
            'consecutive_slow_steps': self.consecutive_slow_steps,
            **reward_stats
        }
        self.metrics_history.append(metrics_log_entry)
        
        log_msg_parts = [
            f"Step {metrics_log_entry['step']}: FPS={metrics_log_entry['fps']:.1f}",
            f"Steps/sec={metrics_log_entry['step_rate']:.1f}",
            f"NaNs(act/inf/ep_met)={self.nan_counts.get('training_actions',0)}/{self.nan_counts.get('training_info',0)}/{self.nan_counts.get('training_episode_metrics',0)}"
        ]
        
        if reward_stats:
            log_msg_parts.extend([
                f"Mean reward (last {len(self.episode_rewards_window)}ep)={metrics_log_entry['mean_reward']:.3f} \u00B1{reward_stats['reward_std']:.3f}",
                f"Trend={reward_stats['reward_trend']:.3f}"
            ])
        if self.verbose > 0: logger.info(", ".join(log_msg_parts))

    def _on_training_end(self) -> None:
        logger.info("DEBUG: EnhancedProgressMonitor._on_training_end called")
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
        
        if self.performance_monitor is not None:
            final_metrics_path = os.path.join(self.performance_monitor.save_dir, "enhanced_monitor_final_metrics.json")
            try:
                with open(final_metrics_path, 'w') as f:
                    json.dump(self.metrics_history, f, indent=4)
                logger.info(f"EnhancedProgressMonitor final metrics history saved to {final_metrics_path}")
            except Exception as e:
                logger.error(f"Failed to save EnhancedProgressMonitor metrics history: {e}")

            if hasattr(self.performance_monitor, 'save_metrics'):
                 self.performance_monitor.save_metrics()
                 logger.info("PerformanceMonitor.save_metrics() called by EnhancedProgressMonitor.")
            else:
                logger.warning("PerformanceMonitor does not have save_metrics attribute.")
        else:
            logger.warning("EnhancedProgressMonitor: self.performance_monitor is None. Cannot save final metrics.")
