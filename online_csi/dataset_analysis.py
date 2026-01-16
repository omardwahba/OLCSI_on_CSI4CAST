"""Dataset Analysis and Visualization Script.

Comprehensive analysis of CSI dataset including:
- Dataset statistics (number of classes, samples, shape)
- Data distribution in real-imaginary domain
- Normalization analysis
- Combined visualization across all conditions

Output: Saved to online_csi/output/dataset_description/
"""

import json
import logging
import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.data_utils import (
    LIST_CHANNEL_MODEL,
    LIST_DELAY_SPREAD,
    LIST_MIN_SPEED_TRAIN,
    CSIDataset,
    _load_data,
)
from src.utils.norm_utils import load_normalization_stats, normalize_input
from src.utils.dirs import DIR_DATA

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """Comprehensive dataset analysis and visualization."""
    
    def __init__(self, output_dir='online_csi/output/dataset_description'):
        """Initialize analyzer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_dataset(self, scenario='TDD', split='train'):
        """Load dataset for analysis.
        
        Args:
            scenario: 'TDD' or 'FDD'
            split: 'train' or 'test'
        
        Returns:
            dict with keys: 'hist', 'pred', 'metadata'
        """
        is_U2D = scenario == 'FDD'
        is_train = split == 'train'
        
        # Create all condition combinations
        conditions = list(product(
            LIST_CHANNEL_MODEL,
            LIST_DELAY_SPREAD,
            LIST_MIN_SPEED_TRAIN
        ))
        
        logger.info(f"Loading {scenario} {split} dataset with {len(conditions)} conditions...")
        
        all_hist = []
        all_pred = []
        all_metadata = []
        
        for cm, ds, ms in conditions:
            try:
                hist = _load_data(
                    dir_data=Path(DIR_DATA),
                    cm=cm,
                    ds=ds,
                    ms=ms,
                    is_train=is_train,
                    is_gen=False,
                    is_hist=True,
                    is_U2D=is_U2D
                )
                
                pred = _load_data(
                    dir_data=Path(DIR_DATA),
                    cm=cm,
                    ds=ds,
                    ms=ms,
                    is_train=is_train,
                    is_gen=False,
                    is_hist=False,
                    is_U2D=is_U2D
                )
                
                if hist is not None and pred is not None:
                    hist_np = hist.numpy() if isinstance(hist, torch.Tensor) else hist
                    pred_np = pred.numpy() if isinstance(pred, torch.Tensor) else pred
                    
                    all_hist.append(hist_np)
                    all_pred.append(pred_np)
                    
                    # Store metadata for each sample
                    for i in range(len(hist_np)):
                        all_metadata.append({
                            'cm': cm,
                            'ds': f"{ds*1e9:.0f}ns",
                            'ms': ms,
                            'condition': f"cm_{cm}_ds_{ds*1e9:.0f}ns_ms_{ms}kmph"
                        })
                    
                    logger.info(f"  Loaded cm={cm}, ds={ds*1e9:.0f}ns, ms={ms}kmph: {len(hist_np)} samples")
            except Exception as e:
                logger.warning(f"Could not load cm={cm}, ds={ds*1e9:.0f}ns, ms={ms}kmph: {e}")
        
        # Concatenate all data
        hist_all = np.concatenate(all_hist, axis=0) if all_hist else None
        pred_all = np.concatenate(all_pred, axis=0) if all_pred else None
        
        if hist_all is not None and pred_all is not None:
            logger.info(f"Total loaded: hist shape={hist_all.shape}, pred shape={pred_all.shape}")
        else:
            logger.error("No data loaded!")
        
        return {
            'hist': hist_all,
            'pred': pred_all,
            'metadata': pd.DataFrame(all_metadata) if all_metadata else pd.DataFrame(),
            'conditions': len(conditions),
            'scenario': scenario,
            'split': split
        }
    
    def compute_statistics(self, data_dict):
        """Compute comprehensive statistics on dataset.
        
        Args:
            data_dict: Dictionary from load_dataset()
        
        Returns:
            dict with statistics
        """
        hist = data_dict['hist']
        pred = data_dict['pred']
        metadata = data_dict['metadata']
        
        stats = {
            'scenario': data_dict['scenario'],
            'split': data_dict['split'],
            'num_conditions': len(metadata['condition'].unique()),
            'num_samples': len(metadata),
            'num_channel_models': len(metadata['cm'].unique()),
            'num_delay_spreads': len(metadata['ds'].unique()),
            'num_speeds': len(metadata['ms'].unique()),
            'channel_models': sorted(metadata['cm'].unique().tolist()),
            'delay_spreads': sorted(metadata['ds'].unique().tolist()),
            'speeds': sorted(metadata['ms'].unique().tolist()),
            'samples_per_condition': metadata.groupby('condition').size().to_dict(),
        }
        
        # Data shape statistics
        stats['hist_shape'] = tuple(hist.shape)
        stats['pred_shape'] = tuple(pred.shape)
        
        # Complex number statistics (real and imaginary parts)
        hist_real = hist.real if np.iscomplexobj(hist) else hist
        hist_imag = hist.imag if np.iscomplexobj(hist) else np.zeros_like(hist)
        pred_real = pred.real if np.iscomplexobj(pred) else pred
        pred_imag = pred.imag if np.iscomplexobj(pred) else np.zeros_like(pred)
        
        # Compute statistics for real and imaginary parts
        for name, (real, imag) in [
            ('hist', (hist_real, hist_imag)),
            ('pred', (pred_real, pred_imag))
        ]:
            # Real part statistics
            stats[f'{name}_real_mean'] = float(np.mean(real))
            stats[f'{name}_real_std'] = float(np.std(real))
            stats[f'{name}_real_min'] = float(np.min(real))
            stats[f'{name}_real_max'] = float(np.max(real))
            
            # Imaginary part statistics
            stats[f'{name}_imag_mean'] = float(np.mean(imag))
            stats[f'{name}_imag_std'] = float(np.std(imag))
            stats[f'{name}_imag_min'] = float(np.min(imag))
            stats[f'{name}_imag_max'] = float(np.max(imag))
            
            # Magnitude statistics
            magnitude = np.abs(real + 1j * imag)
            stats[f'{name}_magnitude_mean'] = float(np.mean(magnitude))
            stats[f'{name}_magnitude_std'] = float(np.std(magnitude))
            stats[f'{name}_magnitude_min'] = float(np.min(magnitude))
            stats[f'{name}_magnitude_max'] = float(np.max(magnitude))
        
        return stats
    
    def create_real_imag_plot(self, data_dict, data_type='hist'):
        """Create real-imaginary scatter plot for all conditions combined.
        
        Args:
            data_dict: Dictionary from load_dataset()
            data_type: 'hist' or 'pred'
        """
        data = data_dict['hist'] if data_type == 'hist' else data_dict['pred']
        metadata = data_dict['metadata']
        
        # Extract real and imaginary parts
        real_parts = data.real if np.iscomplexobj(data) else data
        imag_parts = data.imag if np.iscomplexobj(data) else np.zeros_like(data)
        
        # Flatten to 1D for visualization (sample all subcarriers and antennas)
        real_flat = real_parts.reshape(-1)
        imag_flat = imag_parts.reshape(-1)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(
            f'{data_dict["scenario"]} Dataset - {data_type.upper()} ({data_dict["split"].upper()})\n'
            f'Real-Imaginary Domain Analysis',
            fontsize=16, fontweight='bold'
        )
        
        # 1. Scatter plot (sampled for clarity)
        ax = axes[0, 0]
        # Sample every 100th point for clarity
        sample_idx = np.arange(0, len(real_flat), max(1, len(real_flat) // 10000))
        ax.scatter(real_flat[sample_idx], imag_flat[sample_idx], alpha=0.3, s=1, c='blue')
        ax.set_xlabel('Real Part', fontsize=11, fontweight='bold')
        ax.set_ylabel('Imaginary Part', fontsize=11, fontweight='bold')
        ax.set_title(f'Real-Imaginary Scatter (Sampled {len(sample_idx):,} points)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 2. Histogram of real part
        ax = axes[0, 1]
        ax.hist(real_flat, bins=100, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(np.mean(real_flat), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(real_flat):.4f}')
        ax.axvline(np.median(real_flat), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(real_flat):.4f}')
        ax.set_xlabel('Real Part Value', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Distribution of Real Part', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Histogram of imaginary part
        ax = axes[1, 0]
        ax.hist(imag_flat, bins=100, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(np.mean(imag_flat), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(imag_flat):.4f}')
        ax.axvline(np.median(imag_flat), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(imag_flat):.4f}')
        ax.set_xlabel('Imaginary Part Value', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Distribution of Imaginary Part', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. 2D Histogram (heatmap)
        ax = axes[1, 1]
        h = ax.hist2d(real_flat, imag_flat, bins=100, cmap='viridis')
        plt.colorbar(h[3], ax=ax, label='Count')
        ax.set_xlabel('Real Part', fontsize=11, fontweight='bold')
        ax.set_ylabel('Imaginary Part', fontsize=11, fontweight='bold')
        ax.set_title('2D Density Heatmap', fontsize=12)
        
        plt.tight_layout()
        
        # Save figure
        filename = f'{data_type}_real_imag_{data_dict["scenario"]}_{data_dict["split"]}.png'
        filepath = self.plots_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot: {filepath}")
        plt.close()
    
    def create_per_condition_analysis(self, data_dict):
        """Create visualization showing statistics per condition.
        
        Args:
            data_dict: Dictionary from load_dataset()
        """
        hist = data_dict['hist']
        metadata = data_dict['metadata']
        
        # Extract real parts for analysis
        real_parts = hist.real if np.iscomplexobj(hist) else hist
        imag_parts = hist.imag if np.iscomplexobj(hist) else np.zeros_like(hist)
        
        # Compute statistics per condition
        conditions = metadata['condition'].unique()
        condition_stats = []
        
        for condition in sorted(conditions):
            mask = metadata['condition'] == condition
            condition_idx = np.where(mask)[0]
            
            if len(condition_idx) > 0:
                cond_real = real_parts[condition_idx].flatten()
                cond_imag = imag_parts[condition_idx].flatten()
                
                condition_stats.append({
                    'Condition': condition,
                    'Real Mean': np.mean(cond_real),
                    'Real Std': np.std(cond_real),
                    'Imag Mean': np.mean(cond_imag),
                    'Imag Std': np.std(cond_imag),
                    'Magnitude Mean': np.mean(np.abs(cond_real + 1j * cond_imag)),
                    'Samples': np.sum(mask)
                })
        
        df_stats = pd.DataFrame(condition_stats)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f'{data_dict["scenario"]} Dataset - Per-Condition Statistics ({data_dict["split"].upper()})',
            fontsize=16, fontweight='bold'
        )
        
        # 1. Real part mean by condition
        ax = axes[0, 0]
        ax.bar(range(len(df_stats)), df_stats['Real Mean'], alpha=0.7, color='blue', edgecolor='black')
        ax.set_xticks(range(len(df_stats)))
        ax.set_xticklabels(df_stats['Condition'], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Mean Value', fontsize=11, fontweight='bold')
        ax.set_title('Real Part Mean by Condition', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Real part std by condition
        ax = axes[0, 1]
        ax.bar(range(len(df_stats)), df_stats['Real Std'], alpha=0.7, color='green', edgecolor='black')
        ax.set_xticks(range(len(df_stats)))
        ax.set_xticklabels(df_stats['Condition'], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Std Dev', fontsize=11, fontweight='bold')
        ax.set_title('Real Part Std Dev by Condition', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Imag part mean by condition
        ax = axes[1, 0]
        ax.bar(range(len(df_stats)), df_stats['Imag Mean'], alpha=0.7, color='orange', edgecolor='black')
        ax.set_xticks(range(len(df_stats)))
        ax.set_xticklabels(df_stats['Condition'], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Mean Value', fontsize=11, fontweight='bold')
        ax.set_title('Imaginary Part Mean by Condition', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Magnitude by condition
        ax = axes[1, 1]
        ax.bar(range(len(df_stats)), df_stats['Magnitude Mean'], alpha=0.7, color='red', edgecolor='black')
        ax.set_xticks(range(len(df_stats)))
        ax.set_xticklabels(df_stats['Condition'], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Magnitude Mean', fontsize=11, fontweight='bold')
        ax.set_title('Magnitude Mean by Condition', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save figure
        filename = f'per_condition_stats_{data_dict["scenario"]}_{data_dict["split"]}.png'
        filepath = self.plots_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot: {filepath}")
        plt.close()
        
        # Save statistics to CSV
        csv_filename = f'per_condition_stats_{data_dict["scenario"]}_{data_dict["split"]}.csv'
        csv_filepath = self.output_dir / csv_filename
        df_stats.to_csv(csv_filepath, index=False)
        logger.info(f"Saved CSV: {csv_filepath}")
        
        return df_stats
    
    def create_normalization_analysis(self, data_dict):
        """Analyze if normalization is needed.
        
        Args:
            data_dict: Dictionary from load_dataset()
        """
        hist = data_dict['hist']
        pred = data_dict['pred']
        
        # Compute statistics
        hist_real = hist.real if np.iscomplexobj(hist) else hist
        hist_imag = hist.imag if np.iscomplexobj(hist) else np.zeros_like(hist)
        pred_real = pred.real if np.iscomplexobj(pred) else pred
        pred_imag = pred.imag if np.iscomplexobj(pred) else np.zeros_like(pred)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f'{data_dict["scenario"]} Dataset - Normalization Analysis ({data_dict["split"].upper()})',
            fontsize=16, fontweight='bold'
        )
        
        # 1. Raw histogram with statistics
        ax = axes[0, 0]
        all_real = np.concatenate([hist_real.flatten(), pred_real.flatten()])
        ax.hist(all_real, bins=100, alpha=0.7, color='blue', edgecolor='black')
        mean_val = np.mean(all_real)
        std_val = np.std(all_real)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=2, label=f'±Std: {std_val:.4f}')
        ax.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=2)
        ax.set_xlabel('Real Part Value', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Raw Data Distribution (Real Parts)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Normalized histogram
        ax = axes[0, 1]
        normalized_real = (all_real - mean_val) / std_val
        ax.hist(normalized_real, bins=100, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Normalized Mean: 0')
        ax.axvline(1, color='orange', linestyle='--', linewidth=2, label='±Normalized Std: 1')
        ax.axvline(-1, color='orange', linestyle='--', linewidth=2)
        ax.set_xlabel('Normalized Value', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('After Normalization (Real Parts)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Q-Q plot (check for Gaussian distribution)
        from scipy import stats
        ax = axes[1, 0]
        stats.probplot(all_real, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Gaussian Check)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 4. Statistics table
        ax = axes[1, 1]
        ax.axis('off')
        
        normalization_stats = [
            ['Statistic', 'Raw Data', 'After Norm'],
            ['Mean', f'{mean_val:.6f}', '≈ 0'],
            ['Std Dev', f'{std_val:.6f}', '≈ 1'],
            ['Min', f'{np.min(all_real):.6f}', f'{(np.min(all_real) - mean_val) / std_val:.4f}'],
            ['Max', f'{np.max(all_real):.6f}', f'{(np.max(all_real) - mean_val) / std_val:.4f}'],
            ['Range', f'{np.max(all_real) - np.min(all_real):.4f}', 'Normalized'],
            ['', '', ''],
            ['Normalization Needed?', 'YES - Mean ≠ 0', 'Data centering needed'],
            ['Scaling Needed?', 'YES - Std > 1', 'Variance standardization'],
        ]
        
        table = ax.table(cellText=normalization_stats, cellLoc='left', loc='center',
                        colWidths=[0.3, 0.35, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Normalization Analysis', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save figure
        filename = f'normalization_analysis_{data_dict["scenario"]}_{data_dict["split"]}.png'
        filepath = self.plots_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot: {filepath}")
        plt.close()
    
    def generate_report(self, stats_train, stats_test):
        """Generate comprehensive text report.
        
        Args:
            stats_train: Statistics for training data
            stats_test: Statistics for test data
        """
        report_text = ""
        report_text += "=" * 80 + "\n"
        report_text += "CSI DATASET ANALYSIS REPORT\n"
        report_text += "=" * 80 + "\n\n"
        
        for stats in [stats_train, stats_test]:
            report_text += f"### {stats['scenario']} {stats['split'].upper()} DATASET ###\n\n"
            
            report_text += "DATASET OVERVIEW\n"
            report_text += "-" * 40 + "\n"
            report_text += f"Total Unique Conditions: {stats['num_conditions']}\n"
            report_text += f"Total Samples: {stats['num_samples']}\n"
            report_text += f"Channel Models: {', '.join(stats['channel_models'])} ({stats['num_channel_models']} types)\n"
            report_text += f"Delay Spreads: {', '.join(stats['delay_spreads'])} ({stats['num_delay_spreads']} types)\n"
            report_text += f"Speeds (m/s): {', '.join(map(str, stats['speeds']))} ({stats['num_speeds']} types)\n\n"
            
            report_text += "DATA SHAPE\n"
            report_text += "-" * 40 + "\n"
            report_text += f"Historical CSI Shape: {stats['hist_shape']}\n"
            report_text += f"Prediction Target Shape: {stats['pred_shape']}\n"
            report_text += f"  - Batch Dimension: {stats['hist_shape'][0]}\n"
            report_text += f"  - Antenna/TX Dimension: {stats['hist_shape'][1]}\n"
            report_text += f"  - History/Pred Length: {stats['hist_shape'][2]} / {stats['pred_shape'][2]}\n"
            report_text += f"  - Subcarriers: {stats['hist_shape'][3]}\n\n"
            
            report_text += "NORMALIZATION ANALYSIS\n"
            report_text += "-" * 40 + "\n"
            report_text += f"Historical CSI - Real Part:\n"
            report_text += f"  Mean: {stats['hist_real_mean']:.6f} (should be ≈ 0)\n"
            report_text += f"  Std Dev: {stats['hist_real_std']:.6f} (should be ≈ 1)\n"
            report_text += f"  Range: [{stats['hist_real_min']:.4f}, {stats['hist_real_max']:.4f}]\n\n"
            
            report_text += f"Historical CSI - Imaginary Part:\n"
            report_text += f"  Mean: {stats['hist_imag_mean']:.6f} (should be ≈ 0)\n"
            report_text += f"  Std Dev: {stats['hist_imag_std']:.6f} (should be ≈ 1)\n"
            report_text += f"  Range: [{stats['hist_imag_min']:.4f}, {stats['hist_imag_max']:.4f}]\n\n"
            
            report_text += f"Prediction Target - Real Part:\n"
            report_text += f"  Mean: {stats['pred_real_mean']:.6f} (should be ≈ 0)\n"
            report_text += f"  Std Dev: {stats['pred_real_std']:.6f} (should be ≈ 1)\n"
            report_text += f"  Range: [{stats['pred_real_min']:.4f}, {stats['pred_real_max']:.4f}]\n\n"
            
            report_text += f"Prediction Target - Imaginary Part:\n"
            report_text += f"  Mean: {stats['pred_imag_mean']:.6f} (should be ≈ 0)\n"
            report_text += f"  Std Dev: {stats['pred_imag_std']:.6f} (should be ≈ 1)\n"
            report_text += f"  Range: [{stats['pred_imag_min']:.4f}, {stats['pred_imag_max']:.4f}]\n\n"
            
            report_text += "SAMPLES PER CONDITION\n"
            report_text += "-" * 40 + "\n"
            for cond, count in sorted(stats['samples_per_condition'].items()):
                report_text += f"  {cond}: {count} samples\n"
            report_text += "\n"
            
            report_text += "NORMALIZATION RECOMMENDATION\n"
            report_text += "-" * 40 + "\n"
            
            # Check if normalization is needed
            hist_real_needs_norm = abs(stats['hist_real_mean']) > 0.01 or abs(stats['hist_real_std'] - 1.0) > 0.1
            hist_imag_needs_norm = abs(stats['hist_imag_mean']) > 0.01 or abs(stats['hist_imag_std'] - 1.0) > 0.1
            pred_real_needs_norm = abs(stats['pred_real_mean']) > 0.01 or abs(stats['pred_real_std'] - 1.0) > 0.1
            pred_imag_needs_norm = abs(stats['pred_imag_mean']) > 0.01 or abs(stats['pred_imag_std'] - 1.0) > 0.1
            
            needs_norm = hist_real_needs_norm or hist_imag_needs_norm or pred_real_needs_norm or pred_imag_needs_norm
            
            if needs_norm:
                report_text += "✓ NORMALIZATION REQUIRED\n"
                if hist_real_needs_norm:
                    report_text += f"  - Historical Real Part needs normalization (mean: {stats['hist_real_mean']:.4f}, std: {stats['hist_real_std']:.4f})\n"
                if hist_imag_needs_norm:
                    report_text += f"  - Historical Imaginary Part needs normalization (mean: {stats['hist_imag_mean']:.4f}, std: {stats['hist_imag_std']:.4f})\n"
                if pred_real_needs_norm:
                    report_text += f"  - Prediction Real Part needs normalization (mean: {stats['pred_real_mean']:.4f}, std: {stats['pred_real_std']:.4f})\n"
                if pred_imag_needs_norm:
                    report_text += f"  - Prediction Imaginary Part needs normalization (mean: {stats['pred_imag_mean']:.4f}, std: {stats['pred_imag_std']:.4f})\n"
            else:
                report_text += "✗ Data appears already normalized\n"
            
            report_text += "\n" + "=" * 80 + "\n\n"
        
        return report_text
    
    def run_analysis(self):
        """Run complete analysis pipeline."""
        logger.info("Starting comprehensive dataset analysis...")
        
        all_stats = {}
        
        # Analyze train and test splits for both scenarios
        for scenario in ['TDD', 'FDD']:
            for split in ['train', 'test']:
                logger.info(f"\n=== Analyzing {scenario} {split} ===")
                
                # Load data
                data_dict = self.load_dataset(scenario=scenario, split=split)
                
                if data_dict['hist'] is not None:
                    # Compute statistics
                    stats = self.compute_statistics(data_dict)
                    all_stats[f'{scenario}_{split}'] = stats
                    
                    # Create visualizations
                    self.create_real_imag_plot(data_dict, 'hist')
                    self.create_real_imag_plot(data_dict, 'pred')
                    self.create_per_condition_analysis(data_dict)
                    self.create_normalization_analysis(data_dict)
                    
                    # Log key statistics
                    logger.info(f"Conditions: {stats['num_conditions']}")
                    logger.info(f"Total Samples: {stats['num_samples']}")
                    logger.info(f"Hist Real Mean: {stats['hist_real_mean']:.6f}, Std: {stats['hist_real_std']:.6f}")
                    logger.info(f"Pred Real Mean: {stats['pred_real_mean']:.6f}, Std: {stats['pred_real_std']:.6f}")
        
        # Save all statistics to JSON
        json_path = self.output_dir / 'dataset_statistics.json'
        with open(json_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        logger.info(f"Saved statistics: {json_path}")
        
        # Generate and save report
        stats_train = all_stats.get('TDD_train')
        stats_test = all_stats.get('TDD_test')
        
        if stats_train and stats_test:
            report = self.generate_report(stats_train, stats_test)
            
            report_path = self.output_dir / 'dataset_analysis_report.txt'
            with open(report_path, 'w') as f:
                f.write(report)
            logger.info(f"Saved report: {report_path}")
            
            # Print report to console
            print("\n" + report)
        
        logger.info(f"\n✓ Analysis complete! Results saved to: {self.output_dir}")


def main():
    """Main entry point."""
    analyzer = DatasetAnalyzer()
    analyzer.run_analysis()


if __name__ == '__main__':
    main()
