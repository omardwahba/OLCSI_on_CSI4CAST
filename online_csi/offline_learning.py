"""Offline learning wrapper using CSI-4CAST training framework."""

import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd

from online_csi.config import ExperimentConfig
from online_csi.online_learning import create_online_model

logger = logging.getLogger(__name__)


class OfflineLearner:
    """Offline trainer using standard supervised learning approach."""
    
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        device: torch.device = torch.device("cuda"),
    ):
        """Initialize offline learner."""
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.offline_lr,
            momentum=config.offline_momentum,
            weight_decay=config.offline_weight_decay,
        )
        
        # Setup losses
        from src.cp.loss.loss import NMSELoss, MSELoss, SELoss
        self.criterion_nmse = NMSELoss()
        self.criterion_mse = MSELoss()
        self.criterion_se = SELoss()
    
    def train_epoch(self, train_loader) -> Dict:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_nmse = 0.0
        total_mse = 0.0
        total_se = 0.0
        batch_count = 0
        
        for batch_input, batch_target in train_loader:
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            
            # Forward pass
            pred = self.model(batch_input)
            loss = self.criterion_nmse(pred, batch_target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_nmse += self.criterion_nmse(pred, batch_target).item()
            total_mse += self.criterion_mse(pred, batch_target).item()
            se_pred, se_true = self.criterion_se(pred, batch_target)
            total_se += se_pred.item()
            
            batch_count += 1
            
            if batch_count % 50 == 0:
                logger.info(
                    f"Batch {batch_count}: Loss={loss.item():.6f}, "
                    f"NMSE={total_nmse / batch_count:.6f}"
                )
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        avg_nmse = total_nmse / batch_count if batch_count > 0 else 0
        
        logger.info(f"Epoch complete: Avg Loss={avg_loss:.6f}, Avg NMSE={avg_nmse:.6f}")
        
        return {
            "loss": avg_loss,
            "nmse": avg_nmse,
            "mse": total_mse / batch_count if batch_count > 0 else 0,
            "se": total_se / batch_count if batch_count > 0 else 0,
        }
    
    def evaluate(self, test_loader) -> Tuple[pd.DataFrame, Dict]:
        """Evaluate on test set."""
        self.model.eval()
        
        metrics_list = []
        
        with torch.no_grad():
            for batch_input, batch_target in test_loader:
                batch_input = batch_input.to(self.device)
                batch_target = batch_target.to(self.device)
                
                pred = self.model(batch_input)
                
                # Compute metrics
                nmse = self.criterion_nmse(pred, batch_target).item()
                mse = self.criterion_mse(pred, batch_target).item()
                se_pred, se_true = self.criterion_se(pred, batch_target)
                se = se_pred.item()
                
                metrics_list.append({
                    "nmse": nmse,
                    "mse": mse,
                    "se": se,
                })
        
        df = pd.DataFrame(metrics_list)
        
        summary = {
            "nmse_mean": df["nmse"].mean(),
            "nmse_std": df["nmse"].std(),
            "mse_mean": df["mse"].mean(),
            "mse_std": df["mse"].std(),
            "se_mean": df["se"].mean(),
            "se_std": df["se"].std(),
        }
        
        logger.info(
            f"Test evaluation: NMSE={summary['nmse_mean']:.6f}±{summary['nmse_std']:.6f}, "
            f"SE={summary['se_mean']:.6f}±{summary['se_std']:.6f}"
        )
        
        return df, summary
    
    def train(self, train_loader, test_loader, num_epochs: int = 10) -> Tuple[pd.DataFrame, Dict]:
        """
        Train offline model for multiple epochs.
        
        Returns:
            (test_metrics_df, test_summary)
        """
        logger.info(f"Starting offline training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"\n=== Offline Training Epoch {epoch + 1}/{num_epochs} ===")
            self.train_epoch(train_loader)
        
        # Evaluate on test set
        test_df, test_summary = self.evaluate(test_loader)
        
        return test_df, test_summary
    
    def save_checkpoint(self, checkpoint_path: Path):
        """Save model checkpoint."""
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)
        logger.info(f"Saved offline model: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        logger.info(f"Loaded offline model: {checkpoint_path}")


def create_offline_model(config: ExperimentConfig, device: torch.device) -> nn.Module:
    """Create CNN model for offline training (same as online)."""
    return create_online_model(config, device)
