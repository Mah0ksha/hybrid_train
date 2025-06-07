import numpy as np
import gymnasium as gym
from typing import Dict, Optional, Any, cast
from logger_setup import setup_logger
import time
from trading_environment import TradingEnv
from risk_manager import RiskManager

logger = setup_logger()

class CurriculumWrapper(gym.Wrapper):
    """A curriculum learning wrapper that progressively increases task difficulty"""
    def __init__(self, env: TradingEnv, config: Dict[str, Any]):
        super().__init__(env)
        self.total_timesteps = config.get('total_timesteps', 50000)
        self.current_step = 0
        self.steps_in_stage = 0
        self.current_performance = 0.0
        self.last_improvement_step = 0
        
        # Enhanced stall detection
        self.stall_threshold = config.get('stall_threshold', 50000)  # Steps without improvement
        self.min_performance_change = config.get('min_performance_change', 0.02)
        self.best_performance = float('-inf')
        self.last_performance = 0.0
        self.performance_window = []
        self.window_size = config.get('performance_window_size', 1000)
        
        # Curriculum stages with more granular progression
        self.stages = {
            0: {'name': 'basic', 'volatility_scale': 0.5, 'position_scale': 0.3},
            1: {'name': 'beginner', 'volatility_scale': 0.7, 'position_scale': 0.5},
            2: {'name': 'intermediate', 'volatility_scale': 0.85, 'position_scale': 0.75},
            3: {'name': 'advanced', 'volatility_scale': 1.0, 'position_scale': 1.0}
        }
        self.current_stage = 0
        
        # Performance tracking
        self.success_threshold = config.get('success_threshold', 0.6)  # 60% win rate to advance
        self.eval_window = config.get('eval_window', 100)
        self.trade_outcomes = []
        self.performance_threshold = config.get('performance_threshold', 0.6)
        
        # Progress tracking with enhanced timeout handling
        self.progress_check_frequency = config.get('progress_check_freq', 1000)
        self.last_progress_check = time.time()
        self.progress_timeout = config.get('progress_timeout', 1800)  # 30 minutes default
        self.stage_timeouts = {
            0: config.get('stage_0_timeout', 600),  # 10 minutes for basic
            1: config.get('stage_1_timeout', 1200),  # 20 minutes for beginner
            2: config.get('stage_2_timeout', 1800),  # 30 minutes for intermediate
            3: config.get('stage_3_timeout', 2400)   # 40 minutes for advanced
        }

    def step(self, action):
        env = cast(TradingEnv, self.env)
        original_adaptive_risk_factor = None # Initialize
        
        # Scale risk parameters based on current stage
        if env.risk_manager is not None:
            current_stage_config = self.stages[self.current_stage]
            position_scale = current_stage_config['position_scale']
            original_adaptive_risk_factor = env.risk_manager.adaptive_risk_factor # Store original
            env.risk_manager.adaptive_risk_factor *= position_scale # Apply scaling
        
        # Take step in environment
        obs, reward, done, truncated, info = self.env.step(action)

        # Revert risk_manager adaptive_risk_factor to original value
        if env.risk_manager is not None and original_adaptive_risk_factor is not None:
            env.risk_manager.adaptive_risk_factor = original_adaptive_risk_factor
        
        # Track performance
        if 'trade_result' in info:
            self.trade_outcomes.append(1 if info['trade_result'] > 0 else 0)
            self.trade_outcomes = self.trade_outcomes[-self.eval_window:]
            
            # Update performance tracking
            self.performance_window.append(info['trade_result'])
            if len(self.performance_window) > self.window_size:
                self.performance_window.pop(0)
            
            current_perf = np.mean(self.performance_window) if self.performance_window else 0.0
            if current_perf > self.best_performance:
                self.best_performance = current_perf
                self.last_improvement_step = self.current_step
        
        # Check for stall condition
        if self.current_step - self.last_improvement_step > self.stall_threshold:
            logger.warning(f"Training stalled at step {self.current_step}. No improvement for {self.stall_threshold} steps.")
            # Force stage progression or reset if stalled
            if self.current_stage < len(self.stages) - 1:
                self.current_stage += 1
                self.last_improvement_step = self.current_step  # Reset improvement tracking
                logger.info(f"Forcing progression to stage {self.current_stage} due to stall")
            else:
                # Reset to earlier stage with adjusted parameters
                self.current_stage = max(0, self.current_stage - 1)
                logger.info(f"Resetting to stage {self.current_stage} with adjusted parameters")
        
        # Update curriculum stage
        self._update_stage()
        self.current_step += 1
        
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
          # Apply stage-specific modifications to TradingEnv
        env = cast(TradingEnv, self.env)
        if env.risk_manager is not None:
            current_stage_config = self.stages[self.current_stage]
            volatility_scale = current_stage_config['volatility_scale']
            # Adjust the RiskManager's volatility factor based on curriculum stage
            env.risk_manager.volatility_factor = volatility_scale 
        
        return obs, info
    
    def _update_stage(self):
        """Update curriculum stage based on performance and timing"""
        # Check if we have enough data
        if len(self.trade_outcomes) >= self.eval_window:
            win_rate = np.mean(self.trade_outcomes)
            self.current_performance = win_rate
            
            # Track significant performance changes
            if abs(win_rate - self.last_performance) >= self.min_performance_change:
                self.last_improvement_step = self.current_step
                self.last_performance = win_rate
            
            # Stage progression based on performance
            if (win_rate >= self.performance_threshold and 
                self.current_stage < len(self.stages) - 1):
                self.current_stage += 1
                self.steps_in_stage = 0
                logger.info(f"Performance-based advance to stage {self.current_stage}: "
                          f"{self.stages[self.current_stage]['name']}")
        
        # Time-based checks and stage management
        current_time = time.time()
        time_in_stage = current_time - self.last_progress_check
        
        # Check stage-specific timeouts
        if time_in_stage > self.stage_timeouts[self.current_stage]:
            if self.current_stage < len(self.stages) - 1:
                self.current_stage += 1
                logger.warning(f"Stage {self.current_stage-1} timeout exceeded. Advancing to stage {self.current_stage}")
            else:
                logger.warning("Final stage timeout exceeded. Maintaining current difficulty.")
            self.last_progress_check = current_time
        
        self.steps_in_stage += 1
    
    def get_current_stage(self) -> Dict:
        """Get current curriculum stage settings"""
        return self.stages[self.current_stage]
    
    def should_progress(self) -> bool:
        """Check if we should progress to the next stage"""
        current_time = time.time()
        current_performance = np.mean(self.trade_outcomes) if self.trade_outcomes else 0.0
        
        # Check if we're stuck in a stage too long
        if current_time - self.last_progress_check > self.progress_timeout:
            logger.warning(f"Stage {self.current_stage} exceeded maximum time - forcing progression")
            return True
        
        if self.steps_in_stage % self.progress_check_frequency == 0:
            logger.info(f"Current stage: {self.current_stage}, Steps in stage: {self.steps_in_stage}")
            logger.info(f"Performance: {current_performance:.2f}/{self.performance_threshold}")
            self.last_progress_check = current_time
            
        # Check if performance meets threshold
        return current_performance >= self.performance_threshold
