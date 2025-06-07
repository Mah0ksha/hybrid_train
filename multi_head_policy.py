"""
Modified SB3 policy for multi-head prediction with enhanced architecture
"""
import numpy as np
import torch as th
from torch import nn
import gymnasium as gym
from gymnasium import spaces
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule
from torch.distributions import Normal

from logger_setup import setup_logger
logger = setup_logger()

class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )
        # Add skip connection if dimensions differ
        self.skip = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.block(x) + self.skip(x)

class MultiHeadActorCriticPolicy(ActorCriticPolicy):
    """Enhanced Multi-Head Actor-Critic Policy Network with proper type inheritance"""
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        squash_output: bool = True,
    ):
        if not isinstance(action_space, spaces.Box):
            raise ValueError("MultiHeadActorCriticPolicy only supports continuous action spaces (Box)")
        
        # Convert net_arch to expected format for ActorCriticPolicy
        if net_arch is None:
            # Default architecture if none provided
            processed_net_arch: Dict[str, List[int]] = {"pi": [64, 64], "vf": [64, 64]}
        elif isinstance(net_arch, list): 
            # If a single list is given, use it for both pi and vf networks
            processed_net_arch: Dict[str, List[int]] = {"pi": list(net_arch), "vf": list(net_arch)}
        elif isinstance(net_arch, dict) and "pi" in net_arch and "vf" in net_arch:
            processed_net_arch: Dict[str, List[int]] = net_arch
        else:
            raise ValueError(
                "net_arch must be a list of integers (e.g., [64, 64]) "
                "or a dictionary with keys 'pi' and 'vf' (e.g., {'pi': [64, 64], 'vf': [64, 64]})."
            )
        
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=processed_net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )
        
        self.features_dim = self.features_extractor.features_dim
        action_dim = int(np.prod(action_space.shape))
        
        # Store distribution class
        self.action_dist = Normal
        
        # Initialize variables
        self.log_std = nn.Parameter(th.zeros(action_dim))
        self.action_dim = action_dim
        self.min_std = 1e-6  # Prevent collapse of distribution
        
        # Initialize networks
        self.action_net = nn.Sequential(
            nn.Linear(self.mlp_extractor.latent_dim_pi, action_dim),
            nn.Tanh()
        )

        # Value network (critic)
        self.value_net = nn.Sequential(
            nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        )

        # Risk-aware value network
        self.risk_value_net = nn.Sequential(
            nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        )
        
        # Setup optimizers with improved defaults if none provided
        if optimizer_kwargs is None:
            optimizer_kwargs = {
                'betas': (0.9, 0.999),
                'eps': 1e-5,
                'weight_decay': 1e-4
            }
            
        # Convert learning rate schedule to actual value
        if callable(lr_schedule):
            learning_rate = lr_schedule(1)
        else:
            learning_rate = lr_schedule
            
        optimizer_kwargs['lr'] = learning_rate
        self.optimizer = self.optimizer_class(self.parameters(), **optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)
        
        Args:
            obs: Observation tensor
            deterministic: Whether to sample from action distribution or take mean
            
        Returns:
            actions: Sampled or mean actions
            values: Value function estimates  
            risk_values: Risk-aware value function estimates
        """
        try:
            features = self.extract_features(obs)
            
            # Get latent features from actor and critic networks
            pi_latent = self.mlp_extractor.forward_actor(features)
            vf_latent = self.mlp_extractor.forward_critic(features)
            
            # Get mean actions
            mean_actions = self.action_net(pi_latent)
            
            # Get log standard deviation and clamp to prevent numerical instability
            log_std = self.log_std.clamp(-20, 2)
            std = th.exp(log_std)
            
            # Sample actions from the Gaussian distribution
            if deterministic:
                actions = mean_actions
            else:
                normal_dist = self.action_dist(mean_actions, std)
                actions = normal_dist.rsample()
            
            # Compute values and risk values
            values = self.value_net(vf_latent)
            risk_values = self.risk_value_net(vf_latent)
            
            return actions, values, risk_values
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise
        
    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        """
        Preprocess the observation and extract features
        
        Args:
            obs: Observation tensor
            
        Returns:
            features: Extracted features tensor
        """
        return self.features_extractor(obs)
        
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation
        
        Args:
            observation: Observation tensor
            deterministic: Whether to return deterministic actions
            
        Returns:
            actions: Action tensor
        """
        try:
            actions, _, _ = self.forward(observation, deterministic=deterministic)
            return actions
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise
            
    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy
        
        Args:
            obs: Observation tensor
            actions: Actions tensor to evaluate
            
        Returns:
            values: Value function estimates
            log_prob: Log probabilities of actions
            entropy: Action distribution entropy 
        """
        try:
            features = self.extract_features(obs)
            pi_latent = self.mlp_extractor.forward_actor(features)
            vf_latent = self.mlp_extractor.forward_critic(features)
            
            # Get values from critic
            values = self.value_net(vf_latent)
            
            # Get mean actions from actor
            mean_actions = self.action_net(pi_latent)
            
            # Calculate log probabilities and entropy
            log_std = self.log_std.clamp(-20, 2)
            std = th.exp(log_std)
            
            # Create distribution
            normal_dist = self.action_dist(mean_actions, std)
            log_prob = normal_dist.log_prob(actions).sum(dim=-1)
            entropy = normal_dist.entropy().sum(dim=-1)
            
            return values, log_prob, entropy
            
        except Exception as e:
            logger.error(f"Error in action evaluation: {str(e)}")
            raise
