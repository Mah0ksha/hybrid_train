from dataclasses import dataclass
from typing import Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskConfig:
    # Stop loss and take profit thresholds
    base_stop_loss: float = 0.02  # 2% default stop loss
    base_take_profit: float = 0.03  # 3% default take profit
    max_stop_loss: float = 0.05  # 5% maximum stop loss
    min_stop_loss: float = 0.001  # 0.1% minimum stop loss
    max_take_profit: float = 0.1   # Assuming a max_take_profit exists or is reasonable
    min_take_profit: float = 0.001 # 0.1% minimum take profit
    
    # Position sizing parameters
    max_position_size: float = 0.1  # 10% of portfolio
    min_position_size: float = 0.01  # 1% of portfolio
    risk_per_trade: float = 0.01 # 1% of portfolio value at risk per trade
    transaction_cost: float = 0.001 # 0.1% default transaction cost
    
    # Market volatility thresholds
    high_volatility_threshold: float = 0.02  # 2% daily volatility
    low_volatility_threshold: float = 0.005  # 0.5% daily volatility
    
    # Risk adjustment factors
    volatility_scaling_factor: float = 1.5
    momentum_scaling_factor: float = 1.2
    
    # Portfolio risk limits
    max_portfolio_risk: float = 0.2  # 20% maximum portfolio at risk
    max_correlated_positions: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            key: getattr(self, key)
            for key in self.__annotations__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RiskConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load(cls, filepath: str) -> 'RiskConfig':
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def update(self, **kwargs):
        """Update config parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
    
    def validate(self) -> None:
        """Validate risk configuration parameters."""
        if not (self.min_stop_loss <= self.base_stop_loss <= self.max_stop_loss):
            raise ValueError(f"Invalid stop loss configuration: {self.min_stop_loss} <= {self.base_stop_loss} <= {self.max_stop_loss}")
        
        # Assuming max_take_profit exists or add a reasonable default if it doesn't for validation
        if not hasattr(self, 'max_take_profit'): # Add a default if not present for validation
            self.max_take_profit = 0.2 # Example: 20% max TP if not defined
            logger.warning(f"max_take_profit not defined in RiskConfig, defaulting to {self.max_take_profit} for validation.")

        if not (self.min_take_profit <= self.base_take_profit <= self.max_take_profit):
            raise ValueError(f"Invalid take profit configuration: {self.min_take_profit} <= {self.base_take_profit} <= {self.max_take_profit}")

        if not (0 < self.max_position_size <= 1.0):
            raise ValueError(f"Invalid position size limits: {self.min_position_size} <= {self.max_position_size} <= 1")
        
        # Validate volatility thresholds
        if not (0 < self.low_volatility_threshold < self.high_volatility_threshold):
            raise ValueError(f"Invalid volatility thresholds: {self.low_volatility_threshold} < {self.high_volatility_threshold}")
        
        # Validate scaling factors
        if self.volatility_scaling_factor <= 0:
            raise ValueError(f"volatility_scaling_factor must be positive, got {self.volatility_scaling_factor}")
        if self.momentum_scaling_factor <= 0:
            raise ValueError(f"momentum_scaling_factor must be positive, got {self.momentum_scaling_factor}")
        
        # Validate portfolio limits
        if not (0 < self.max_portfolio_risk < 1):
            raise ValueError(f"max_portfolio_risk must be between 0 and 1, got {self.max_portfolio_risk}")
        if self.max_correlated_positions < 1:
            raise ValueError(f"max_correlated_positions must be positive, got {self.max_correlated_positions}")

        if not (0 <= self.transaction_cost < 0.1): # Example: transaction cost should be non-negative and less than 10%
            raise ValueError(f"Invalid transaction_cost: {self.transaction_cost}. Must be >= 0 and < 0.1")
