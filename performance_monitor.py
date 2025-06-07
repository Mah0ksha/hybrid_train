import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
import traceback
from collections import deque
from typing import Dict, Any

from logger_setup import setup_logger
from trading_metrics import EnhancedTradingMetrics

logger = setup_logger()

class PerformanceMonitor:
    def __init__(self, save_dir='checkpoints'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.metrics_calculator = EnhancedTradingMetrics()
        
        # Performance tracking
        self.rolling_window = 100
        self.trade_history = deque(maxlen=self.rolling_window)
        
        # Real-time metrics
        self.current_metrics = {}

    def update_trade_metrics(self, trade_info: Dict[str, Any]):
        """Update metrics with new trade information"""
        try:
            self.trade_history.append(trade_info)
            self.metrics_calculator.process_episode_trades(list(self.trade_history))
            self.current_metrics = self.metrics_calculator.calculate_performance_metrics()
            
            # Store detailed trade metrics
            self.save_metrics()

        except Exception as e:
            logger.error(f"Error updating trade metrics: {str(e)}")
            logger.error(traceback.format_exc())
    
    def update_training_metrics(self, epoch, train_metrics, val_metrics):
        """Update training metrics after each epoch"""
        try:
            # This functionality can be handled separately or integrated with a different system
            # For now, we focus on trade metrics
            pass
            
        except Exception as e:
            logger.error(f"Error updating training metrics: {str(e)}")
            logger.error(traceback.format_exc())

    def save_metrics(self):
        """Save current metrics to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            metrics_file = os.path.join(self.save_dir, f'metrics_{timestamp}.json')
            
            metrics_data = {
                'current_metrics': self.current_metrics,
                'trade_history': list(self.trade_history)
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, default=str)
                
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            logger.error(traceback.format_exc())
    
    def get_summary(self):
        """Get current performance summary"""
        return {
            'current_metrics': self.current_metrics,
            'recent_trades': len(self.trade_history),
        }
