"""Online learning trainer for CSI prediction with per-batch updates."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

from src.cp.models.cnn_online import CNNOnline
from src.cp.loss.loss import NMSELoss, MSELoss, SELoss
from online_csi.config import ExperimentConfig

logger = logging.getLogger(__name__)


class OnlineCNNTrainer:
    """Online trainer that updates model on every batch (batch_size=1 or small)."""
    
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        optimizer: Optional[optim.Optimizer] = None,
        device: torch.device = torch.device("cuda"),
    ):
        """
        Initialize online trainer.
        
        Args:
            model: Neural network model
            config: Experiment configuration
            optimizer: Optimizer (created if None)
            device: Computation device
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Setup optimizer
        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr=config.online_lr)
        self.optimizer = optimizer
        
        # Setup losses
        self.criterion_nmse = NMSELoss()
        self.criterion_mse = MSELoss()
        self.criterion_se = SELoss()
        
        # Metrics tracking
        self.train_metrics = []
        self.test_metrics = []
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        phase: str = "train",
        max_batches: Optional[int] = None,
    ) -> List[Dict]:
        """
        Run one online training epoch (per-batch updates).
        
        Args:
            train_loader: Data loader yielding (input, target, metadata)
            phase: "train" (always update) or "test" (evaluate only)
            max_batches: Maximum number of batches (None = all)
        
        Returns:
            List of per-batch metric dicts with keys:
            {batch_idx, cm, ds, ms, nmse, mse, se, loss, time_ms}
        """
        self.model.train()
        
        batch_metrics = []
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, (batch_input, batch_target, metadata_list) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            
            batch_start = time.perf_counter()
            
            # Forward pass
            with torch.set_grad_enabled(phase == "train"):
                pred = self.model(batch_input)
                loss = self.criterion_nmse(pred, batch_target)
                
                # Backward pass
                if phase == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            
            batch_end = time.perf_counter()
            batch_time_ms = (batch_end - batch_start) * 1000
            
            # Compute metrics
            nmse = self.criterion_nmse(pred, batch_target).item()
            mse = self.criterion_mse(pred, batch_target).item()
            se_pred, se_true = self.criterion_se(pred, batch_target)
            se = se_pred.item()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Log metrics for each sample in batch
            for sample_idx, metadata in enumerate(metadata_list):
                metric_dict = {
                    "batch_idx": batch_idx,
                    "sample_idx": sample_idx,
                    "cm": metadata["cm"],
                    "ds": metadata["ds"],
                    "ms": metadata["ms"],
                    "nmse": nmse,
                    "mse": mse,
                    "se": se,
                    "loss": loss.item(),
                    "time_ms": batch_time_ms,
                }
                batch_metrics.append(metric_dict)
            
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / batch_count
                logger.info(
                    f"{phase.upper()} Batch {batch_idx + 1}: "
                    f"Loss={avg_loss:.6f}, NMSE={nmse:.6f}, SE={se:.6f}"
                )
        
        logger.info(
            f"{phase.upper()} epoch complete: "
            f"Avg Loss={total_loss / batch_count:.6f}, "
            f"Total batches={batch_count}"
        )
        
        return batch_metrics
    
    def evaluate(
        self,
        test_loader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> List[Dict]:
        """
        Evaluate model on test set (no gradient updates).
        
        Args:
            test_loader: Test data loader
            max_batches: Maximum batches to evaluate (None = all)
        
        Returns:
            List of per-batch metric dicts
        """
        self.model.eval()
        
        test_metrics = []
        
        with torch.no_grad():
            for batch_idx, (batch_input, batch_target, metadata_list) in enumerate(test_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                batch_input = batch_input.to(self.device)
                batch_target = batch_target.to(self.device)
                
                pred = self.model(batch_input)
                
                # Compute metrics
                nmse = self.criterion_nmse(pred, batch_target).item()
                mse = self.criterion_mse(pred, batch_target).item()
                se_pred, se_true = self.criterion_se(pred, batch_target)
                se = se_pred.item()
                loss = self.criterion_nmse(pred, batch_target).item()
                
                # Log metrics for each sample
                for sample_idx, metadata in enumerate(metadata_list):
                    metric_dict = {
                        "batch_idx": batch_idx,
                        "sample_idx": sample_idx,
                        "cm": metadata["cm"],
                        "ds": metadata["ds"],
                        "ms": metadata["ms"],
                        "nmse": nmse,
                        "mse": mse,
                        "se": se,
                        "loss": loss,
                    }
                    test_metrics.append(metric_dict)
                
                if (batch_idx + 1) % 100 == 0:
                    logger.info(f"Test Batch {batch_idx + 1}: NMSE={nmse:.6f}, SE={se:.6f}")
        
        logger.info(f"Test evaluation complete: {len(test_metrics)} samples")
        
        return test_metrics
    
    def train_and_test(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int = 1,
        max_batches: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run complete online training loop: alternately train and test.
        
        For online learning, model is updated on both train AND test data.
        This simulates real-world adaptive deployment.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            num_epochs: Number of training epochs
            max_batches: Max batches per epoch (None = all)
        
        Returns:
            (train_metrics_df, test_metrics_df)
        """
        all_train_metrics = []
        all_test_metrics = []
        
        for epoch in range(num_epochs):
            logger.info(f"\n=== Online Training Epoch {epoch + 1}/{num_epochs} ===")
            
            # Training phase (with updates)
            logger.info("Training phase...")
            train_metrics = self.train_epoch(
                train_loader,
                phase="train",
                max_batches=max_batches,
            )
            all_train_metrics.extend(train_metrics)
            
            # Testing phase (ALSO with updates for online learning)
            logger.info("Testing phase (with updates)...")
            test_metrics = self.train_epoch(
                test_loader,
                phase="test",
                max_batches=max_batches,
            )
            all_test_metrics.extend(test_metrics)
        
        # Convert to DataFrames
        train_df = pd.DataFrame(all_train_metrics)
        test_df = pd.DataFrame(all_test_metrics)
        
        logger.info(f"\nOnline training complete: {len(train_df)} train, {len(test_df)} test samples")
        
        return train_df, test_df
    
    def save_checkpoint(self, checkpoint_path: Path):
        """Save model checkpoint."""
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        logger.info(f"Loaded checkpoint: {checkpoint_path}")


def create_online_model(config: ExperimentConfig, device: torch.device) -> nn.Module:
    """
    Create CNN model for online learning.
    
    Note: This creates the raw model, not the PyTorch Lightning wrapper.
    """
    from src.cp.models.cnn_online import CNNOnlineModel
    
    model = CNNOnlineModel(
        num_subcarriers=config.num_subcarriers,
        hist_len=config.hist_len,
        pred_len=config.pred_len,
        cnn_hidden_dims=config.cnn_hidden_dims,
        mlp_hidden_dim=config.mlp_hidden_dim,
    )
    
    return model.to(device)
