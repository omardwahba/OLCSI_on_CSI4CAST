"""Visualization utilities for online/offline comparison."""

import logging
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

logger = logging.getLogger(__name__)


def setup_plotting():
    """Setup matplotlib style."""
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 10


def plot_error_evolution(
    metrics_df: pd.DataFrame,
    output_path: Path,
    metric_name: str = "nmse",
    title: str = "Error Evolution",
    phase: str = "test",
):
    """
    Plot error evolution vs batch index, colored by condition.
    
    Args:
        metrics_df: Metrics dataframe with batch_idx, cm, ds, ms, and metric column
        output_path: Where to save the plot
        metric_name: Which metric to plot (nmse, mse, se)
        title: Plot title
        phase: "train" or "test"
    """
    if "batch_idx" not in metrics_df.columns or metric_name not in metrics_df.columns:
        logger.warning(f"Skipping error evolution plot: missing columns")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Get unique conditions
    if "cm" in metrics_df.columns:
        conditions = metrics_df.groupby(["cm", "ds", "ms"]).ngroups
        
        # Plot each condition with different color
        for (cm, ds, ms), group in metrics_df.groupby(["cm", "ds", "ms"]):
            label = f"CM {cm} DS {ds}ns MS {ms}"
            ax.scatter(
                group["batch_idx"],
                group[metric_name],
                label=label,
                alpha=0.6,
                s=20,
            )
    else:
        ax.plot(metrics_df["batch_idx"], metrics_df[metric_name], alpha=0.7)
    
    # Add rolling average
    rolling = metrics_df[metric_name].rolling(window=100, min_periods=1).mean()
    ax.plot(
        metrics_df["batch_idx"],
        rolling,
        color="red",
        linewidth=2,
        label="Rolling Avg (window=100)",
        alpha=0.8,
    )
    
    ax.set_xlabel("Batch Index")
    ax.set_ylabel(metric_name.upper())
    ax.set_title(f"{title} - {phase.upper()}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved error evolution plot: {output_path}")


def plot_loss_trajectory(
    online_train_df: pd.DataFrame,
    online_test_df: pd.DataFrame,
    output_path: Path,
):
    """Plot training and test loss trajectory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Compute rolling averages
    if "batch_idx" in online_train_df.columns and len(online_train_df) > 0:
        train_rolling = online_train_df["loss"].rolling(window=100, min_periods=1).mean()
        ax.plot(
            online_train_df["batch_idx"],
            train_rolling,
            label="Train Loss (rolling avg)",
            linewidth=2,
            color="blue",
            alpha=0.7,
        )
    
    if "batch_idx" in online_test_df.columns and len(online_test_df) > 0:
        test_rolling = online_test_df["loss"].rolling(window=100, min_periods=1).mean()
        ax.plot(
            online_test_df["batch_idx"],
            test_rolling,
            label="Test Loss (rolling avg)",
            linewidth=2,
            color="orange",
            alpha=0.7,
        )
    
    ax.set_xlabel("Batch Index")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Test Loss Trajectory (Online Learning)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Saved loss trajectory plot: {output_path}")


def plot_metric_comparison(
    online_test_df: pd.DataFrame,
    offline_test_df: pd.DataFrame,
    output_path: Path,
    metric: str = "nmse",
):
    """
    Create box plots comparing online vs offline metrics.
    
    Args:
        online_test_df: Online test metrics
        offline_test_df: Offline test metrics
        output_path: Where to save plot
        metric: Which metric to compare (nmse, mse, se)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    data_to_plot = []
    labels = []
    
    if metric in online_test_df.columns:
        data_to_plot.append(online_test_df[metric].values)
        labels.append("Online")
    
    if metric in offline_test_df.columns:
        data_to_plot.append(offline_test_df[metric].values)
        labels.append("Offline")
    
    # Create box plot
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # Color boxes
    colors = ["lightblue", "lightgreen"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} Comparison: Online vs Offline")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Saved comparison plot: {output_path}")


def plot_adaptation_heatmap(
    metrics_df: pd.DataFrame,
    output_path: Path,
    metric: str = "nmse",
):
    """
    Create heatmap showing error vs batch index vs condition.
    
    This shows which conditions require longer adaptation.
    """
    if "cm" not in metrics_df.columns or "batch_idx" not in metrics_df.columns:
        logger.warning("Cannot create adaptation heatmap: missing columns")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create condition labels
    metrics_df["condition"] = (
        "CM" + metrics_df["cm"].astype(str) +
        "_DS" + metrics_df["ds"].astype(str) +
        "_MS" + metrics_df["ms"].astype(str)
    )
    
    # Pivot table: conditions vs batch index bins vs metric
    batch_bins = pd.cut(metrics_df["batch_idx"], bins=20)
    pivot_data = metrics_df.pivot_table(
        values=metric,
        index="condition",
        columns=batch_bins,
        aggfunc="mean",
    )
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    sns.heatmap(
        pivot_data,
        cmap="RdYlGn_r",
        cbar_kws={"label": metric.upper()},
        ax=ax,
    )
    
    ax.set_xlabel("Batch Index (binned)")
    ax.set_ylabel("Channel Condition")
    ax.set_title(f"Error Evolution by Condition - {metric.upper()}")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved adaptation heatmap: {output_path}")


def plot_all_metrics(
    online_train_df: pd.DataFrame,
    online_test_df: pd.DataFrame,
    offline_test_df: pd.DataFrame,
    output_dir: Path,
):
    """Generate all visualization plots."""
    setup_plotting()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Error evolution plots
    if not online_train_df.empty:
        plot_error_evolution(
            online_train_df,
            output_dir / "error_evolution_train.png",
            metric_name="nmse",
            title="NMSE Evolution (Training)",
            phase="train",
        )
        
        plot_error_evolution(
            online_train_df,
            output_dir / "mse_evolution_train.png",
            metric_name="mse",
            title="MSE Evolution (Training)",
            phase="train",
        )
        
        plot_error_evolution(
            online_train_df,
            output_dir / "se_evolution_train.png",
            metric_name="se",
            title="SE Evolution (Training)",
            phase="train",
        )
    
    if not online_test_df.empty:
        plot_error_evolution(
            online_test_df,
            output_dir / "error_evolution_test.png",
            metric_name="nmse",
            title="NMSE Evolution (Testing)",
            phase="test",
        )
        
        plot_error_evolution(
            online_test_df,
            output_dir / "mse_evolution_test.png",
            metric_name="mse",
            title="MSE Evolution (Testing)",
            phase="test",
        )
        
        plot_error_evolution(
            online_test_df,
            output_dir / "se_evolution_test.png",
            metric_name="se",
            title="SE Evolution (Testing)",
            phase="test",
        )
    
    # Loss trajectory
    if not online_train_df.empty and not online_test_df.empty:
        plot_loss_trajectory(online_train_df, online_test_df, output_dir / "loss_trajectory.png")
    
    # Comparison plots
    if not online_test_df.empty and not offline_test_df.empty:
        plot_metric_comparison(
            online_test_df,
            offline_test_df,
            output_dir / "nmse_comparison_boxplot.png",
            metric="nmse",
        )
        
        plot_metric_comparison(
            online_test_df,
            offline_test_df,
            output_dir / "se_comparison_boxplot.png",
            metric="se",
        )
    
    # Adaptation heatmap
    if not online_test_df.empty:
        plot_adaptation_heatmap(
            online_test_df,
            output_dir / "adaptation_by_condition.png",
            metric="nmse",
        )
    
    logger.info(f"Generated all plots in {output_dir}")
