import unittest
import numpy as np
from trading_environment import TradingEnvironment
from multi_head_policy import MultiHeadActorCriticPolicy
from curriculum_learning import CurriculumWrapper
from performance_monitor import PerformanceMonitor
from risk_manager import RiskManager

class TestIntegratedSystem(unittest.TestCase):
    def setUp(self):
        self.env = TradingEnvironment()
        self.risk_manager = RiskManager()
        self.performance_monitor = PerformanceMonitor()
        
    def test_multi_head_policy_outputs(self):
        policy = MultiHeadActorCriticPolicy(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space
        )
        obs = self.env.reset()
        actions = policy.forward(obs)
        self.assertEqual(len(actions), 3)  # Three heads
        
    def test_curriculum_progression(self):
        wrapper = CurriculumWrapper(self.env)
        initial_difficulty = wrapper.current_difficulty
        wrapper.update_difficulty(success_rate=0.8)
        self.assertGreater(wrapper.current_difficulty, initial_difficulty)
        
    def test_risk_management_adaptation(self):
        high_volatility_state = np.array([0.5, 2.0, 1.5])  # Mock market state
        risk_params = self.risk_manager.adjust_risk_parameters(high_volatility_state)
        self.assertIn('stop_loss', risk_params)
        self.assertIn('take_profit', risk_params)
        
    def test_performance_metrics(self):
        returns = [0.01, -0.02, 0.03, 0.01, -0.01]
        self.performance_monitor.update(returns)
        metrics = self.performance_monitor.get_metrics()
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('daily_pnl', metrics)

if __name__ == '__main__':
    unittest.main()
