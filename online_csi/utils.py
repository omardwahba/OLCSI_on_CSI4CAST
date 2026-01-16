"""Utility functions for online/offline learning experiments."""

import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.data_utils import (
    load_data,
    CSIDataset,
    collect_fn_separate_antennas,
)
from src.utils.norm_utils import load_normalization_stats, normalize_input
from online_csi.config import ExperimentConfig

logger = logging.getLogger(__name__)


class UnifiedCSIDataset(Dataset):
    """Unified dataset combining all channel conditions with metadata tracking."""
    
    def __init__(self, h_hist: torch.Tensor, h_pred: torch.Tensor, metadata: List[Dict]):
        """
        Initialize unified CSI dataset.
        
        Args:
            h_hist: Historical CSI [total_samples, 32, 16, 300] complex or [total_samples*32, 16, 600] real
            h_pred: Prediction CSI [total_samples, 32, 4, 300] complex or [total_samples*32, 4, 600] real
            metadata: List of dicts with keys {cm, ds, ms} for each sample
        """
        self.h_hist = h_hist
        self.h_pred = h_pred
        self.metadata = metadata
        
        assert len(h_hist) == len(h_pred), "hist and pred must have same length"
        assert len(h_hist) == len(metadata), "metadata must match data length"
    
    def __len__(self):
        return len(self.h_hist)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Return (hist, pred, metadata) for index."""
        return self.h_hist[idx], self.h_pred[idx], self.metadata[idx]


def auto_discover_conditions(config: ExperimentConfig) -> Dict[str, List]:
    """
    Auto-discover available channel conditions in data directory.
    
    Returns:
        Dict with keys {cm, ds, ms} and lists of available values
    """
    data_root = config.data_root / "train" / "regular"
    
    conditions = {"cm": [], "ds": [], "ms": []}
    
    if not data_root.exists():
        logger.warning(f"Data root does not exist: {data_root}")
        return conditions
    
    for folder in data_root.iterdir():
        if not folder.is_dir():
            continue
        
        # Parse folder name: cm_X_ds_YYY_ms_ZZ
        parts = folder.name.split("_")
        if len(parts) >= 6:
            try:
                cm = parts[1]
                ds = int(parts[3])
                ms = int(parts[5])
                
                if cm not in conditions["cm"]:
                    conditions["cm"].append(cm)
                if ds not in conditions["ds"]:
                    conditions["ds"].append(ds)
                if ms not in conditions["ms"]:
                    conditions["ms"].append(ms)
            except (ValueError, IndexError):
                continue
    
    # Sort for consistency
    conditions["cm"] = sorted(conditions["cm"])
    conditions["ds"] = sorted(conditions["ds"])
    conditions["ms"] = sorted(conditions["ms"])
    
    logger.info(f"Auto-discovered conditions: {conditions}")
    return conditions


def load_unified_dataset(
    config: ExperimentConfig,
    is_train: bool = True,
    is_normalized: bool = True,
) -> Tuple[UnifiedCSIDataset, Dict]:
    """
    Load and combine all available channel conditions into single unified dataset.
    
    Args:
        config: Experiment configuration
        is_train: Load training (True) or test (False) data
        is_normalized: Apply normalization after loading
    
    Returns:
        (UnifiedCSIDataset, stats_dict)
    """
    # Determine is_U2D based on scenario
    is_U2D = (config.scenario == "FDD")
    
    # Auto-discover available conditions
    conditions = auto_discover_conditions(config)
    
    if not conditions["cm"]:
        raise ValueError(f"No data found in {config.data_root}")
    
    logger.info(f"Loading {'train' if is_train else 'test'} data for {config.scenario}")
    logger.info(f"Combining {len(conditions['cm'])} CMs × {len(conditions['ds'])} DSs × {len(conditions['ms'])} speeds")
    
    all_hist = []
    all_pred = []
    all_metadata = []
    
    sample_count = 0
    
    # Load data for each condition
    for cm in conditions["cm"]:
        for ds in conditions["ds"]:
            for ms in conditions["ms"]:
                try:
                    # Convert delay spread from ns to float
                    ds_float = float(ds) * 1e-9
                    
                    # Load hist and pred
                    h_hist = load_data(
                        dir_data=config.data_root,
                        list_cm=[cm],
                        list_ds=[ds_float],
                        list_ms=[ms],
                        is_train=is_train,
                        is_gen=False,
                        is_hist=True,
                        is_U2D=is_U2D,
                    )
                    
                    h_pred = load_data(
                        dir_data=config.data_root,
                        list_cm=[cm],
                        list_ds=[ds_float],
                        list_ms=[ms],
                        is_train=is_train,
                        is_gen=False,
                        is_hist=False,
                        is_U2D=is_U2D,
                    )
                    
                    if h_hist is None or h_pred is None:
                        logger.warning(f"Missing data for cm={cm}, ds={ds}ns, ms={ms}")
                        continue
                    
                    all_hist.append(h_hist)
                    all_pred.append(h_pred)
                    
                    # Track metadata for each sample
                    n_samples = h_hist.shape[0]
                    for _ in range(n_samples):
                        all_metadata.append({
                            "cm": cm,
                            "ds": ds,
                            "ms": ms,
                            "is_U2D": is_U2D,
                        })
                    
                    sample_count += n_samples
                    logger.debug(f"Loaded cm={cm}, ds={ds}ns, ms={ms}: {n_samples} samples")
                    
                except Exception as e:
                    logger.warning(f"Error loading cm={cm}, ds={ds}ns, ms={ms}: {e}")
                    continue
    
    if not all_hist:
        raise ValueError("No data loaded. Check data path and conditions.")
    
    # Concatenate all data
    h_hist_all = torch.cat(all_hist, dim=0)
    h_pred_all = torch.cat(all_pred, dim=0)
    
    logger.info(f"Total samples loaded: {sample_count}")
    logger.info(f"Concatenated shapes: hist={h_hist_all.shape}, pred={h_pred_all.shape}")
    
    # Apply normalization if requested
    if is_normalized:
        try:
            stats = load_normalization_stats(config.data_root, is_U2D=is_U2D)
            h_hist_all, h_pred_all = normalize_input(h_hist_all, h_pred_all, is_U2D=is_U2D)
            logger.info("Normalization applied")
            stats_dict = {"normalized": True, "stats": stats}
        except Exception as e:
            logger.warning(f"Normalization failed: {e}. Using unnormalized data.")
            stats_dict = {"normalized": False}
    else:
        stats_dict = {"normalized": False}
    
    # Create dataset
    dataset = UnifiedCSIDataset(h_hist_all, h_pred_all, all_metadata)
    
    logger.info(f"Created UnifiedCSIDataset with {len(dataset)} total samples")
    
    return dataset, stats_dict


def get_data_loaders(
    config: ExperimentConfig,
    train_dataset: Optional[UnifiedCSIDataset] = None,
    test_dataset: Optional[UnifiedCSIDataset] = None,
    online_batch_size: int = 1,
    offline_batch_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for online and offline training/testing.
    
    Args:
        config: Experiment configuration
        train_dataset: Training dataset (loaded if None)
        test_dataset: Test dataset (loaded if None)
        online_batch_size: Batch size for online learning (default 1)
        offline_batch_size: Batch size for offline learning (default from config)
    
    Returns:
        (online_train_loader, online_test_loader, offline_train_loader, offline_test_loader)
    """
    if offline_batch_size is None:
        offline_batch_size = config.offline_batch_size
    
    # Load datasets if not provided
    if train_dataset is None:
        train_dataset, _ = load_unified_dataset(config, is_train=True)
    if test_dataset is None:
        test_dataset, _ = load_unified_dataset(config, is_train=False)
    
    # Create loaders
    # Online: batch_size=1, no shuffling (preserve order for error evolution)
    online_train_loader = DataLoader(
        train_dataset,
        batch_size=online_batch_size,
        shuffle=False,
        collate_fn=_online_collate_fn,
    )
    
    online_test_loader = DataLoader(
        test_dataset,
        batch_size=online_batch_size,
        shuffle=False,
        collate_fn=_online_collate_fn,
    )
    
    # Offline: standard batch size, with shuffling
    offline_train_loader = DataLoader(
        train_dataset,
        batch_size=offline_batch_size,
        shuffle=True,
        collate_fn=_offline_collate_fn,
    )
    
    offline_test_loader = DataLoader(
        test_dataset,
        batch_size=offline_batch_size,
        shuffle=False,
        collate_fn=_offline_collate_fn,
    )
    
    logger.info(
        f"Created loaders: "
        f"online_train={len(online_train_loader)}, "
        f"online_test={len(online_test_loader)}, "
        f"offline_train={len(offline_train_loader)}, "
        f"offline_test={len(offline_test_loader)}"
    )
    
    return online_train_loader, online_test_loader, offline_train_loader, offline_test_loader


def _online_collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """
    Collate function for online learning that preserves metadata.
    
    Converts complex to real format using separate antenna processing.
    """
    hist_list = []
    pred_list = []
    metadata_list = []
    
    for hist, pred, metadata in batch:
        hist_list.append(hist)
        pred_list.append(pred)
        metadata_list.append(metadata)
    
    # Stack batches
    hist = torch.stack(hist_list, dim=0)  # [batch_size, 32, 16, 300]
    pred = torch.stack(pred_list, dim=0)   # [batch_size, 32, 4, 300]
    
    # Convert to separate antenna format (real-valued)
    # [batch_size, 32, 16, 300] -> [batch_size*32, 16, 600]
    hist = hist.view(-1, hist.shape[2], hist.shape[3])  # [batch*32, 16, 300]
    pred = pred.view(-1, pred.shape[2], pred.shape[3])  # [batch*32, 4, 300]
    
    # Convert complex to real
    hist = torch.view_as_real(hist)  # [batch*32, 16, 300, 2]
    pred = torch.view_as_real(pred)  # [batch*32, 4, 300, 2]
    
    hist = hist.view(hist.shape[0], hist.shape[1], -1)  # [batch*32, 16, 600]
    pred = pred.view(pred.shape[0], pred.shape[1], -1)  # [batch*32, 4, 600]
    
    return hist, pred, metadata_list


def _offline_collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for offline learning (standard CSI-4CAST style).
    Uses the framework's collect_fn_separate_antennas.
    """
    # Repackage batch as tuples without metadata
    batch_no_metadata = [(hist, pred) for hist, pred, _ in batch]
    
    # Use framework's collate function
    return collect_fn_separate_antennas(batch_no_metadata)


def setup_logging(config: ExperimentConfig, log_file: Optional[str] = None):
    """Setup logging for experiment."""
    config.create_output_dirs()
    
    if log_file is None:
        log_file = str(config.logs_dir / "experiment.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    
    logger.info(f"Logging setup. Output directory: {config.output_dir}")
