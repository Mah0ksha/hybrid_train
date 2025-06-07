# Standard library imports
from collections import defaultdict
import sys
import time
import traceback
import os
import glob
from typing import Dict, List, Optional, Union, Sequence, Any, Tuple
import argparse # For command-line arguments
import json # For logging config
import copy # For deepcopying config

# Third-party imports
import numpy as np
import torch
# import torch.nn as nn # No longer directly needed here if old Pytorch model is removed
# from torch.cuda import amp # No longer directly needed here
# from torch.utils.data import Dataset # No longer directly needed here
import pandas as pd
# import torch.nn.functional as F # No longer directly needed here
# from sklearn.metrics import f1_score, recall_score # No longer directly needed here

# Local imports
from logger_setup import setup_logger
from hybrid_trainer import HybridTrainer # Import the new HybridTrainer
from multi_head_policy import MultiHeadActorCriticPolicy # Import for policy selection

# Import configuration constants (many are used by HybridTrainer via its config)
from config_vals import (
    START_CAP, TRANSACTION_COST, MAX_DRAWDOWN, RISK_PER_TRADE, STOP_LOSS,
    MAX_POSITION_SIZE, TRAILING_STOP_PERCENT, WINDOW_SIZE,
    MAX_TIMESTEPS, # Default for total_timesteps if not overridden by arg
    # RL_TIMESTEPS, # Might be superseded by args.total_timesteps for HybridTrainer
    N_STEPS, N_EPOCHS, BATCH_SIZE, LEARNING_RATE, GAMMA, 
    ENTROPY_COEF, VF_COEF, MAX_GRAD_NORM, REWARD_SCALE
    # PROGRESS_INTERVAL, # Was for old PyTorch loop
    # CURRENT_UTC_TIME, UNIVERSE_FILE, RAW_DIR, # These seem environment/data setup specific, ensure data_path arg is used
)

# Import SL training function
from supervised_pretrainer import train_sl_model, WINDOW_SIZE as SL_WINDOW_SIZE, NUM_FEATURES as SL_NUM_FEATURES

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# RL_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl_models") # HybridTrainer configures its own save path

def run_hybrid_trainer_pipeline(config: dict, logger):
    logger.info(f"Initializing HybridTrainer for {config.get('target_symbol', 'all_symbols')} with config: {config}")
    trainer = HybridTrainer(config=config, logger=logger)
    results = trainer.run_training_pipeline()
    logger.info(f"Training pipeline finished for symbol: {config.get('target_symbol', 'all_symbols')}. Results: {results}")
    return results

def main():
    logger = setup_logger()
    parser = argparse.ArgumentParser(description="Hybrid RL/SL Trading Bot Training Framework")

    # Execution Mode
    parser.add_argument("--mode", type=str, default="rl_train", choices=["sl_train", "rl_train", "full_pipeline"],
                        help="Execution mode: sl_train (only SL), rl_train (RL, possibly w/ pre-training), full_pipeline (SL then RL)")

    # Common paths and general RL config
    parser.add_argument("--data_path", type=str, default=None, help="Path to main data file (e.g., sample_ohlcv.csv or for single symbol runs if raw_data_dir is not used).")
    parser.add_argument("--universe_path", type=str, default="universe.csv", help="Path to the universe CSV file for multi-symbol training.")
    parser.add_argument("--raw_data_dir_path", type=str, default="raw_data_a/", help="Directory containing raw per-symbol CSV files (e.g., SYMBOL_1min.csv).")
    parser.add_argument("--model_save_path_base", type=str, default="rl_models/hybrid_ppo", help="Base path to save RL models (symbol and suffix will be added).")
    parser.add_argument("--log_dir_base", type=str, default="logs/per_symbol_runs/", help="Base directory for logs (symbol-specific subdirs will be created).")
    parser.add_argument("--total_timesteps", type=int, default=MAX_TIMESTEPS, help="Total timesteps for RL training.")
    parser.add_argument("--eval_freq", type=int, default=5000, help="Evaluation frequency for BacktestCallback.")
    parser.add_argument("--window_size", type=int, default=WINDOW_SIZE, help="Observation window size for RL.")
    parser.add_argument("--policy", type=str, default="MlpPolicy", help="Policy network for RL model.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for PPO.")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate for PPO.")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="Discount factor for PPO.")
    parser.add_argument("--n_steps", type=int, default=N_STEPS, help="N-steps for PPO.")
    parser.add_argument("--n_epochs", type=int, default=N_EPOCHS, help="Number of epochs for PPO update.")
    parser.add_argument("--ent_coef", type=float, default=ENTROPY_COEF, help="Entropy coefficient for PPO.")
    parser.add_argument("--vf_coef", type=float, default=VF_COEF, help="Value function coefficient for PPO.")
    parser.add_argument("--max_grad_norm", type=float, default=MAX_GRAD_NORM, help="Max gradient norm for PPO.")
    parser.add_argument("--reward_scale", type=float, default=REWARD_SCALE, help="Reward scale for TradingEnv.")
    parser.add_argument("--transaction_cost_pct", type=float, default=TRANSACTION_COST, help="Transaction cost percentage for TradingEnv.")
    parser.add_argument("--stop_loss_penalty", type=float, default=1.0, help="Penalty for hitting stop loss.")
    parser.add_argument("--profit_incentive", type=float, default=0.1, help="Bonus reward for profitable trades, as a percentage of PnL.")
    parser.add_argument("--ppo_verbose", type=int, default=1, choices=[0, 1, 2], help="PPO verbosity level.")
    parser.add_argument("--device", type=str, default="auto", help="Device for PyTorch ('cpu', 'cuda', 'auto').")

    # Risk Manager Parameters (prefix with rm_ for clarity in config)
    parser.add_argument("--rm_max_position_size", type=float, default=MAX_POSITION_SIZE, help="RiskManager: Max position size as fraction of portfolio.")
    parser.add_argument("--rm_max_drawdown", type=float, default=MAX_DRAWDOWN, help="RiskManager: Max drawdown percentage.")
    parser.add_argument("--rm_stop_loss_pct", type=float, default=STOP_LOSS, help="RiskManager: Stop loss percentage.")
    parser.add_argument("--rm_risk_per_trade", type=float, default=RISK_PER_TRADE, help="RiskManager: Risk per trade as fraction of portfolio.")
    parser.add_argument("--rm_transaction_cost", type=float, default=TRANSACTION_COST, help="RiskManager: Transaction cost for its calculations (can differ from env).")
    parser.add_argument("--rm_take_profit_pct", type=float, default=0.04, help="RiskManager: Take profit percentage.")
    parser.add_argument("--rm_trailing_stop_pct", type=float, default=TRAILING_STOP_PERCENT, help="RiskManager: Trailing stop percentage.")

    # SL Pre-training Parameters
    parser.add_argument("--processed_data_path_for_sl", type=str, default="data/processed_with_sl_labels.csv", 
                        help="Path to data file with indicators and SL labels, for SL training.")
    parser.add_argument("--sl_model_save_path", type=str, default="sl_models/sl_policy_pretrain.pth", 
                        help="Path to save the SL trained model.")
    parser.add_argument("--sl_model_load_path", type=str, default="sl_models/sl_policy_pretrain.pth", 
                        help="Path to load a pre-trained SL model for RL fine-tuning.")
    parser.add_argument("--use_sl_pretraining", action='store_true', help="Enable SL pre-training for RL agent.")
    parser.add_argument("--sl_epochs", type=int, default=20, help="Epochs for SL model training.")
    parser.add_argument("--sl_lr", type=float, default=1e-3, help="Learning rate for SL model training.")
    parser.add_argument("--sl_batch_size", type=int, default=64, help="Batch size for SL model training.")
    parser.add_argument("--sl_policy_hidden_dim_1", type=int, default=256, help="Size of the first hidden layer in SLPolicyNetwork and PPO policy_net.")
    parser.add_argument("--sl_policy_hidden_dim_2", type=int, default=256, help="Size of the second hidden layer in SLPolicyNetwork and PPO policy_net.")
    # SL_WINDOW_SIZE and SL_NUM_FEATURES are imported from supervised_pretrainer and should match its defaults/usage

    # New training strategy arguments
    parser.add_argument("--use_lr_scheduler", action='store_true', help="Enable learning rate scheduler during RL training.")
    parser.add_argument("--use_curriculum_learning", action='store_true', help="Enable curriculum learning during RL training.")

    args = parser.parse_args()

    # --- SL Training Mode ---
    if args.mode == "sl_train" or args.mode == "full_pipeline":
        logger.info("==== Starting Supervised Learning (SL) Model Training ====")
        if not os.path.exists(args.processed_data_path_for_sl):
            logger.error(f"SL training data not found: {args.processed_data_path_for_sl}. \nPlease ensure preprocess_data.py has been run with generate_sl_labels=True and output saved here.")
            if args.mode == "full_pipeline":
                logger.error("Cannot proceed to RL training in full_pipeline mode without SL model.")
            return # Exit if data not found

        sl_save_dir = os.path.dirname(args.sl_model_save_path)
        if sl_save_dir and not os.path.exists(sl_save_dir):
            os.makedirs(sl_save_dir, exist_ok=True)

        train_sl_model(
            processed_data_path=args.processed_data_path_for_sl,
            model_save_path=args.sl_model_save_path,
            window_size=SL_WINDOW_SIZE, # Use imported constant
            num_features=SL_NUM_FEATURES, # Use imported constant
            epochs=args.sl_epochs,
            batch_size=args.sl_batch_size,
            learning_rate=args.sl_lr,
            # test_size is default in train_sl_model
        )
        logger.info(f"==== SL Model Training Completed. Model saved to: {args.sl_model_save_path} ====")
        if args.mode == "sl_train":
            return # Exit if only SL training was requested
    
    # --- RL Training Mode (or continuation of full_pipeline) ---
    logger.info("==== Starting Reinforcement Learning (RL) Hybrid Training ====")

    # Construct base config from args for HybridTrainer
    # This config will be further specialized per symbol if universe.csv is used
    base_hybrid_config = {
        "data_path": args.data_path, # For single file or dummy data fallback
        "universe_path": args.universe_path,
        "raw_data_dir_path": args.raw_data_dir_path,
        "model_save_path": args.model_save_path_base, # Base path, symbol will be appended by trainer
        # log_dir will be set per symbol later
        "total_timesteps": args.total_timesteps,
        "eval_freq": args.eval_freq,
        "window_size": args.window_size, 
        "transaction_cost_pct": args.transaction_cost_pct,
        "reward_scale": args.reward_scale,
        "stop_loss_penalty": args.stop_loss_penalty,
        "profit_incentive": args.profit_incentive,
        "policy": args.policy,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "n_steps": args.n_steps,
        "n_epochs": args.n_epochs,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "ppo_verbose": args.ppo_verbose,
        "device": args.device,
        
        # RiskManager params
        "max_position_size": args.rm_max_position_size,
        "max_drawdown": args.rm_max_drawdown,
        "stop_loss_pct": args.rm_stop_loss_pct,
        "risk_per_trade": args.rm_risk_per_trade,
        "transaction_cost": args.rm_transaction_cost, # Note: distinct from env's transaction_cost_pct
        "take_profit_pct": args.rm_take_profit_pct,
        "trailing_stop_pct": args.rm_trailing_stop_pct,
        
        # RiskConfig internal details (can be same as RiskManager for simplicity)
        "base_stop_loss": args.rm_stop_loss_pct, 
        "base_take_profit": args.rm_take_profit_pct,
        "max_risk_config_position_size": args.rm_max_position_size,

        # Default technical indicators (can be overridden in a config file later if needed)
        "initial_indicators": [
            "rsi", 
            "rsi_28",
            "macd", 
            "bollinger", 
            "atr", 
            "vwap",
            "sma_50", 
            "sma_200",
            "price_div_sma_50",
            "std_dev_14"
        ],
        
        # Env step counts
        "train_max_steps": args.total_timesteps, # Default train env max_steps to total_timesteps for one episode
        "val_max_steps": args.total_timesteps // 4, # Example
        "test_max_steps": args.total_timesteps // 2, # Example

        # Curriculum and evaluation flags
        "use_curriculum": args.use_curriculum_learning,
        "evaluate_on_validation_set": True,
        "evaluate_on_test_set": True,
        "progress_bar": True, # For model.learn()
        "use_lr_scheduler": args.use_lr_scheduler,

        # SL Pre-training specific args for HybridTrainer
        "use_sl_pretraining": args.use_sl_pretraining if args.mode == "rl_train" else (True if args.mode == "full_pipeline" else False),
        "sl_model_load_path": args.sl_model_save_path if args.mode == "full_pipeline" else (args.sl_model_load_path if args.use_sl_pretraining else None),
        "sl_policy_net_arch_pi": [args.sl_policy_hidden_dim_1, args.sl_policy_hidden_dim_2],
        "sl_policy_net_arch_vf": [64, 64] # Default for PPO value function, can be configured if needed
    }

    # Read symbols from universe.csv to run training per symbol
    if os.path.exists(args.universe_path):
        try:
            symbols_df = pd.read_csv(args.universe_path)
            if 'symbol' not in symbols_df.columns:
                logger.error(f"Universe file {args.universe_path} must contain a 'symbol' column.")
                symbols_to_train = [None] # Fallback to single run mode
            else:
                symbols_to_train = symbols_df['symbol'].astype(str).tolist()
                logger.info(f"Found symbols in universe file to train: {symbols_to_train}")
        except Exception as e:
            logger.error(f"Error reading universe file {args.universe_path}: {e}. Defaulting to single run mode (no symbol).")
            symbols_to_train = [None] # Represents a single run without a specific symbol from universe
    else:
        logger.warning(f"Universe file {args.universe_path} not found. Defaulting to single run mode (no symbol).")
        symbols_to_train = [None]

    all_run_results = {}

    for symbol in symbols_to_train:
        symbol_config = base_hybrid_config.copy()
        if symbol:
            logger.info(f"==== Starting training pipeline for symbol: {symbol} ====")
            symbol_config["target_symbol"] = symbol
            symbol_config["log_dir"] = os.path.join(args.log_dir_base, symbol)
            symbol_config["model_save_path"] = f"{args.model_save_path_base}_{symbol}.zip"
        else: # Single run mode
            logger.info("==== Starting training pipeline for single run (no specific symbol) ====")
            symbol_config["log_dir"] = args.log_dir_base
            symbol_config["model_save_path"] = f"{args.model_save_path_base}.zip"

        try:
            results = run_hybrid_trainer_pipeline(config=symbol_config, logger=logger)
            all_run_results[symbol if symbol else 'single_run'] = results
        except Exception as e:
            logger.error(f"--- FAILED TRAINING FOR SYMBOL: {symbol} ---")
            logger.error(f"Error: {e}")
            logger.error(traceback.format_exc())
            all_run_results[symbol if symbol else 'single_run'] = {"status": "failed", "error": str(e)}

    logger.info("==== All Training Runs Completed ====")
    # Optionally, save a summary of all results
    summary_path = os.path.join(args.log_dir_base, "all_runs_summary.json")
    try:
        # A simple conversion for summary - more robust serialization may be needed
        serializable_results = json.loads(json.dumps(all_run_results, default=str))
        with open(summary_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        logger.info(f"Full run summary saved to {summary_path}")
    except Exception as e:
        logger.error(f"Could not save all_runs_summary.json. Error: {e}")

if __name__ == "__main__":
    main()
