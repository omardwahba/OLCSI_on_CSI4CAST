#!/usr/bin/env python3
"""
Main experiment runner for online vs offline CSI prediction comparison.

Usage:
    python online_csi/run_experiment.py --scenario TDD
    python online_csi/run_experiment.py --scenario FDD
    
    OR from project root:
    python -m online_csi.run_experiment --scenario TDD
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from datetime import datetime

import torch

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from online_csi.config import ExperimentConfig
from online_csi.utils import (
    setup_logging,
    load_unified_dataset,
    get_data_loaders,
)
from online_csi.online_learning import OnlineCNNTrainer, create_online_model
from online_csi.offline_learning import OfflineLearner, create_offline_model
from online_csi.evaluation import save_metrics, compare_results, save_comparison
from online_csi.visualization import plot_all_metrics

logger = logging.getLogger(__name__)


def run_offline_training(
    config: ExperimentConfig,
    train_loader,
    test_loader,
) -> tuple:
    """
    Train offline model using standard supervised learning.
    
    Returns:
        (model, test_metrics_df, test_summary)
    """
    logger.info("\n" + "="*80)
    logger.info("STARTING OFFLINE TRAINING")
    logger.info("="*80)
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    model = create_offline_model(config, device)
    logger.info(f"Created offline model: {config.model_name}")
    
    # Create trainer
    learner = OfflineLearner(model, config, device=device)
    
    # Train
    logger.info(f"Training for {config.offline_epochs} epochs...")
    test_df, test_summary = learner.train(train_loader, test_loader, config.offline_epochs)
    
    # Save model
    offline_model_path = config.models_dir / "offline_model_final.pt"
    learner.save_checkpoint(offline_model_path)
    
    logger.info("\n" + "="*80)
    logger.info("OFFLINE TRAINING COMPLETE")
    logger.info(f"Test NMSE: {test_summary['nmse_mean']:.6f} ± {test_summary['nmse_std']:.6f}")
    logger.info(f"Test SE:   {test_summary['se_mean']:.6f} ± {test_summary['se_std']:.6f}")
    logger.info("="*80)
    
    return model, test_df, test_summary


def run_online_training(
    config: ExperimentConfig,
    train_loader,
    test_loader,
    offline_model: torch.nn.Module = None,
) -> tuple:
    """
    Train online model with per-batch updates on both train and test data.
    
    Args:
        config: Experiment configuration
        train_loader: Training data loader
        test_loader: Test data loader
        offline_model: Pre-trained offline model to initialize from (optional)
    
    Returns:
        (model, train_metrics_df, test_metrics_df)
    """
    logger.info("\n" + "="*80)
    logger.info("STARTING ONLINE TRAINING & TESTING")
    logger.info("="*80)
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    model = create_online_model(config, device)
    logger.info(f"Created online model: {config.model_name}")
    
    # Initialize from offline model if provided
    if offline_model is not None:
        try:
            model.load_state_dict(offline_model.state_dict())
            logger.info("Initialized online model from offline model weights")
        except Exception as e:
            logger.warning(f"Could not load offline weights: {e}. Using random init.")
    
    # Create trainer
    trainer = OnlineCNNTrainer(model, config, device=device)
    logger.info(f"Learning rate: {config.online_lr}")
    
    # Run online training (model updates on both train and test data)
    logger.info("Running online training with per-batch updates...")
    logger.info("Note: Model will be updated on BOTH training and test data")
    train_df, test_df = trainer.train_and_test(
        train_loader,
        test_loader,
        num_epochs=1,  # Single epoch (process all data once)
        max_batches=None,  # Use all data
    )
    
    # Save model
    online_model_path = config.models_dir / "online_model_final.pt"
    trainer.save_checkpoint(online_model_path)
    
    logger.info("\n" + "="*80)
    logger.info("ONLINE TRAINING COMPLETE")
    if not train_df.empty:
        logger.info(f"Training:  {len(train_df)} samples processed")
    if not test_df.empty:
        logger.info(f"Testing:   {len(test_df)} samples processed")
        logger.info(f"Final Test NMSE: {test_df['nmse'].iloc[-1]:.6f}")
    logger.info("="*80)
    
    return model, train_df, test_df


def save_experiment_config(config: ExperimentConfig, output_path: Path):
    """Save experiment configuration to JSON."""
    config_dict = {
        "scenario": config.scenario,
        "device": config.device,
        "model": config.model_name,
        "num_subcarriers": config.num_subcarriers,
        "hist_len": config.hist_len,
        "pred_len": config.pred_len,
        "is_separate_antennas": config.is_separate_antennas,
        "online_lr": config.online_lr,
        "offline_lr": config.offline_lr,
        "offline_epochs": config.offline_epochs,
        "offline_batch_size": config.offline_batch_size,
        "timestamp": datetime.now().isoformat(),
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"Saved experiment config: {output_path}")


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(
        description="Online vs Offline CSI Prediction Comparison"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["TDD", "FDD"],
        required=True,
        help="Scenario: TDD or FDD",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Computation device",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Run only offline training (skip online)",
    )
    parser.add_argument(
        "--online-only",
        action="store_true",
        help="Run only online training (skip offline)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: limit data to first 1000 samples",
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = ExperimentConfig(
        scenario=args.scenario,
        device=args.device if torch.cuda.is_available() else "cpu",
    )
    
    # Setup logging
    setup_logging(config, log_file=str(config.logs_dir / "experiment.log"))
    
    logger.info("\n" + "="*80)
    logger.info("CSI PREDICTION: ONLINE vs OFFLINE COMPARISON")
    logger.info("="*80)
    logger.info(f"Scenario: {config.scenario}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Output: {config.output_dir}")
    logger.info("="*80)
    
    try:
        # Load unified datasets
        logger.info("\nLoading unified datasets...")
        train_dataset, train_stats = load_unified_dataset(config, is_train=True)
        test_dataset, test_stats = load_unified_dataset(config, is_train=False)
        
        # Create data loaders
        logger.info("Creating data loaders...")
        online_train_loader, online_test_loader, offline_train_loader, offline_test_loader = get_data_loaders(
            config,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )
        
        offline_model = None
        offline_test_df = None
        offline_summary = None
        online_train_df = None
        online_test_df = None
        
        # Run offline training
        if not args.online_only:
            offline_model, offline_test_df, offline_summary = run_offline_training(
                config,
                offline_train_loader,
                offline_test_loader,
            )
            
            # Save offline metrics
            save_metrics(
                offline_test_df,
                config.metrics_dir / "offline_metrics.csv",
                "offline",
                config,
            )
        
        # Run online training
        if not args.offline_only:
            online_model, online_train_df, online_test_df = run_online_training(
                config,
                online_train_loader,
                online_test_loader,
                offline_model=offline_model,
            )
            
            # Save online metrics
            if not online_train_df.empty:
                save_metrics(
                    online_train_df,
                    config.metrics_dir / "online_metrics.csv",
                    "online_train",
                    config,
                )
            
            if not online_test_df.empty:
                save_metrics(
                    online_test_df,
                    config.metrics_dir / "online_metrics.csv",
                    "online_test",
                    config,
                )
        
        # Comparison
        if offline_test_df is not None and online_test_df is not None:
            logger.info("\nGenerating comparison analysis...")
            comparison = compare_results(
                online_train_df if online_train_df is not None else pd.DataFrame(),
                online_test_df,
                offline_test_df,
                config,
            )
            save_comparison(comparison, config.metrics_dir / "comparison_summary.json")
        
        # Visualization
        logger.info("\nGenerating visualizations...")
        plot_all_metrics(
            online_train_df if online_train_df is not None else pd.DataFrame(),
            online_test_df if online_test_df is not None else pd.DataFrame(),
            offline_test_df if offline_test_df is not None else pd.DataFrame(),
            config.plots_dir,
        )
        
        # Save experiment config
        save_experiment_config(config, config.output_dir / "experiment_config.json")
        
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT COMPLETE!")
        logger.info(f"Results saved to: {config.output_dir}")
        logger.info("="*80)
        
        return 0
    
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import pandas as pd  # Import here to avoid import errors
    sys.exit(main())
