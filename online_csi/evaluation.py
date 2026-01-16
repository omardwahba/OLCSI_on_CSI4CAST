"""Evaluation framework for comparing online and offline results."""

import logging
import json
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np

from online_csi.config import ExperimentConfig

logger = logging.getLogger(__name__)


def aggregate_metrics(metrics_df: pd.DataFrame) -> Dict:
    """Aggregate metrics from per-batch/per-sample dataframe."""
    
    summary = {
        "nmse_mean": float(metrics_df["nmse"].mean()),
        "nmse_std": float(metrics_df["nmse"].std()),
        "nmse_min": float(metrics_df["nmse"].min()),
        "nmse_max": float(metrics_df["nmse"].max()),
        
        "mse_mean": float(metrics_df["mse"].mean()),
        "mse_std": float(metrics_df["mse"].std()),
        "mse_min": float(metrics_df["mse"].min()),
        "mse_max": float(metrics_df["mse"].max()),
        
        "se_mean": float(metrics_df["se"].mean()),
        "se_std": float(metrics_df["se"].std()),
        "se_min": float(metrics_df["se"].min()),
        "se_max": float(metrics_df["se"].max()),
        
        "total_samples": len(metrics_df),
    }
    
    # Add timing info if available
    if "time_ms" in metrics_df.columns:
        summary["time_ms_mean"] = float(metrics_df["time_ms"].mean())
        summary["time_ms_std"] = float(metrics_df["time_ms"].std())
    
    return summary


def compute_convergence_time(metrics_df: pd.DataFrame, threshold_ratio: float = 0.9) -> int:
    """
    Estimate convergence time as when error stabilizes.
    
    Args:
        metrics_df: Metrics dataframe with batch_idx and nmse columns
        threshold_ratio: When rolling mean reaches 90% of final value
    
    Returns:
        Batch index when converged (or -1 if not converged)
    """
    if "batch_idx" not in metrics_df.columns:
        return -1
    
    # Compute rolling mean (window=100)
    rolling_nmse = metrics_df["nmse"].rolling(window=100, min_periods=1).mean()
    
    # Find convergence point
    final_nmse = rolling_nmse.iloc[-1]
    threshold = final_nmse / threshold_ratio
    
    convergence_idx = (rolling_nmse <= threshold).idxmax()
    
    if convergence_idx == 0:
        return -1  # Not converged
    
    return int(metrics_df.loc[convergence_idx, "batch_idx"])


def per_condition_analysis(metrics_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Analyze metrics grouped by channel condition (cm, ds, ms).
    
    Returns:
        Dict mapping condition strings to metric dicts
    """
    if "cm" not in metrics_df.columns:
        return {}
    
    analysis = {}
    
    for cm in metrics_df["cm"].unique():
        cm_data = metrics_df[metrics_df["cm"] == cm]
        
        for ds in cm_data["ds"].unique():
            ds_data = cm_data[cm_data["ds"] == ds]
            
            for ms in ds_data["ms"].unique():
                ms_data = ds_data[ds_data["ms"] == ms]
                
                condition_key = f"CM_{cm}_ds_{ds}_ms_{ms}"
                analysis[condition_key] = {
                    "nmse_mean": float(ms_data["nmse"].mean()),
                    "nmse_std": float(ms_data["nmse"].std()),
                    "mse_mean": float(ms_data["mse"].mean()),
                    "se_mean": float(ms_data["se"].mean()),
                    "samples": len(ms_data),
                }
    
    return analysis


def save_metrics(
    metrics_df: pd.DataFrame,
    output_path: Path,
    name: str,
    config: ExperimentConfig,
):
    """Save metrics to CSV and JSON summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save detailed CSV
    csv_path = output_path.with_name(f"{name}_detailed.csv")
    metrics_df.to_csv(csv_path, index=False)
    logger.info(f"Saved detailed metrics: {csv_path}")
    
    # Save summary JSON
    summary = aggregate_metrics(metrics_df)
    
    # Add per-condition analysis if available
    if "cm" in metrics_df.columns:
        summary["per_condition"] = per_condition_analysis(metrics_df)
    
    # Add convergence time if batch_idx available
    if "batch_idx" in metrics_df.columns:
        summary["convergence_batch"] = compute_convergence_time(metrics_df)
    
    json_path = output_path.with_name(f"{name}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary: {json_path}")
    
    return summary


def compare_results(
    online_train_df: pd.DataFrame,
    online_test_df: pd.DataFrame,
    offline_test_df: pd.DataFrame,
    config: ExperimentConfig,
) -> Dict:
    """
    Compare online vs offline results.
    
    Returns:
        Comparison summary dict
    """
    online_train_summary = aggregate_metrics(online_train_df)
    online_test_summary = aggregate_metrics(online_test_df)
    offline_test_summary = aggregate_metrics(offline_test_df)
    
    comparison = {
        "scenario": config.scenario,
        "timestamp": pd.Timestamp.now().isoformat(),
        
        "online": {
            "train": online_train_summary,
            "test": online_test_summary,
        },
        
        "offline": {
            "test": offline_test_summary,
        },
        
        "comparison": {
            "nmse_improvement": float(
                (offline_test_summary["nmse_mean"] - online_test_summary["nmse_mean"]) 
                / offline_test_summary["nmse_mean"] * 100
            ),
            "se_improvement": float(
                (online_test_summary["se_mean"] - offline_test_summary["se_mean"]) 
                / offline_test_summary["se_mean"] * 100
            ),
        },
    }
    
    logger.info("\n=== COMPARISON RESULTS ===")
    logger.info(f"Online NMSE:  {online_test_summary['nmse_mean']:.6f}")
    logger.info(f"Offline NMSE: {offline_test_summary['nmse_mean']:.6f}")
    logger.info(f"NMSE Improvement: {comparison['comparison']['nmse_improvement']:.2f}%")
    logger.info(f"SE Improvement: {comparison['comparison']['se_improvement']:.2f}%")
    
    return comparison


def save_comparison(comparison: Dict, output_path: Path):
    """Save comparison results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Saved comparison: {output_path}")
