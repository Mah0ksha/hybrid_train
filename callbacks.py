from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import time
import numpy as np
import torch
from logger_setup import setup_logger
from stable_baselines3.common.vec_env import VecEnv

logger = setup_logger()

# Simple progress callback
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

# Learning rate scheduler with warmup
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

# Backtest callback
class BacktestCallback(EvalCallback):
    def __init__(self, eval_env, eval_freq=1000, deterministic=False):
        super().__init__(eval_env, eval_freq=eval_freq, deterministic=deterministic)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
        return True

class CustomEvalCallback(BaseCallback):
    """
    Custom callback for evaluating the agent during training.
    This callback records detailed trading metrics from the custom environment.
    """
    def __init__(self, eval_env: VecEnv, eval_freq: int, deterministic: bool = False):
        super().__init__(verbose=0)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.training_evaluation_history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            obs = self.eval_env.reset()
            dones = [False]
            while not dones[0]:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, rewards, dones, infos = self.eval_env.step(action)
            
            # The environment is automatically reset when done
            # We can now get the metrics from the completed episode
            metrics = self.eval_env.env_method('get_metrics')[0]
            if metrics:
                metrics['timestep'] = self.n_calls
                self.training_evaluation_history.append(metrics)
                logger.info(f"CustomEvalCallback logged metrics at step {self.n_calls}: {metrics}")
        return True

class LrSchedulerCallback(BaseCallback):
    def __init__(self, initial_lr, final_lr, total_timesteps, power=1.0):
        super().__init__()
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_timesteps = total_timesteps
        self.power = power

    def _on_step(self):
        fraction = 1.0 - (self.num_timesteps / self.total_timesteps)
        lr = self.final_lr + (self.initial_lr - self.final_lr) * (fraction ** self.power)
        
        # Access the optimizer from the model
        if self.model and hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
            self.optimizer = self.model.policy.optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return True
