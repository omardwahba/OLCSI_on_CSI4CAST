# Quick Start Guide - Online Learning CSI Prediction

## 5-Minute Quick Start

### Run FDD Scenario
```bash
cd /home/omar/csi/OLCSI_on_CSI4CAST
python online_csi/run_experiment.py --scenario FDD
```

### Run TDD Scenario
```bash
python online_csi/run_experiment.py --scenario TDD
```

Both scenarios will:
1. ✅ Load unified dataset (27 conditions auto-discovered)
2. ✅ Train offline model for 10 epochs
3. ✅ Run online training with per-batch updates
4. ✅ Generate 9 visualization plots
5. ✅ Save metrics to CSV/JSON
6. ✅ Display comparison results

**Execution time**: ~2 minutes per scenario on GPU, ~5 minutes on CPU

**Output location**: `online_csi/output/{fdd|tdd}/`

---

## File Overview

### Configuration & Setup
- **[config.py](./config.py)** - Experiment configuration (scenarios, learning rates, paths)
- **[utils.py](./utils.py)** - Data loading and unified dataset creation

### Training & Evaluation
- **[online_learning.py](./online_learning.py)** - Online trainer (per-batch updates)
- **[offline_learning.py](./offline_learning.py)** - Offline trainer (batch learning)
- **[evaluation.py](./evaluation.py)** - Metrics computation and comparison

### Visualization & Execution
- **[visualization.py](./visualization.py)** - Plot generation (9 different plots)
- **[run_experiment.py](./run_experiment.py)** - Main pipeline orchestration

### Documentation
- **[RESULTS_SUMMARY.md](./RESULTS_SUMMARY.md)** - Detailed results and findings
- **[IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md)** - Complete technical documentation
- **[README.md](./README.md)** - This file

---

## Key Results

### FDD Scenario Results
```
Offline Test NMSE: 1.002602 ± 0.001626
Online Test NMSE:  1.000739 (after adaptation)
Improvement:       0.19% (NMSE), 1.71% (SE)
```

### TDD Scenario Results
```
Offline Test NMSE: 1.002735 ± 0.000010
Online Test NMSE:  1.000506 (after adaptation)
Improvement:       0.22% (NMSE), 0.42% (SE)
```

---

## Command Options

### Scenario Selection
```bash
--scenario FDD|TDD    # Which scenario to run (default: FDD)
```

### Device Selection
```bash
--device cuda|cpu     # GPU or CPU (default: cuda if available)
```

### Training Modes
```bash
--offline-only       # Skip online training (testing offline only)
--online-only        # Skip offline training (requires pre-trained model)
```

### Debug & Logging
```bash
--debug             # Enable verbose logging for troubleshooting
```

### Examples
```bash
# Run both offline and online on GPU (default)
python online_csi/run_experiment.py --scenario FDD

# Run on CPU
python online_csi/run_experiment.py --scenario FDD --device cpu

# Run offline only
python online_csi/run_experiment.py --scenario FDD --offline-only

# Run online only (requires offline model)
python online_csi/run_experiment.py --scenario FDD --online-only

# Run with debug output
python online_csi/run_experiment.py --scenario FDD --debug
```

---

## Output Files

### Metrics (Auto-saved)
```
online_csi/output/fdd/metrics/
├── offline_detailed.csv           # Offline training metrics
├── offline_summary.json           # Offline statistics
├── online_train_detailed.csv      # Online training per-batch
├── online_train_summary.json      # Training statistics
├── online_test_detailed.csv       # Online testing per-batch
├── online_test_summary.json       # Testing statistics
└── comparison_summary.json        # Online vs Offline comparison
```

### Plots (Auto-saved)
```
online_csi/output/fdd/plots/
├── error_evolution_train.png      # NMSE scatter plot
├── error_evolution_test.png       # Test error evolution
├── mse_evolution_*.png            # Raw MSE values
├── se_evolution_*.png             # Spectral efficiency
├── loss_trajectory.png            # Train/test loss
├── nmse_comparison_boxplot.png    # Offline vs online
├── se_comparison_boxplot.png      # SE comparison
└── adaptation_by_condition.png    # Per-condition heatmap
```

### Models (Auto-saved)
```
online_csi/output/fdd/models/
├── offline_model_final.pt         # Trained offline model
└── online_model_final.pt          # Online-adapted model
```

---

## Data Format

### Unified Dataset
- **27 conditions**: 3 CMs (A, C, D) × 3 DSs (30, 100, 300 ns) × 3 Speeds (1, 10, 30 kmph)
- **54 training samples**: One sample per condition
- **54 test samples**: One sample per condition
- **Auto-discovery**: Automatically finds all available conditions

### Per-Sample Metadata
Each sample includes:
- `cm`: Channel Model (A/C/D)
- `ds`: Delay Spread in ns (30/100/300)
- `ms`: Max Speed in kmph (1/10/30)
- `is_U2D`: Uplink-to-Downlink flag (TDD/FDD)

This metadata is preserved in CSV outputs for error evolution analysis.

---

## Performance Metrics Explained

### NMSE (Normalized Mean Squared Error)
- **Definition**: ||predicted - actual||² / ||actual||²
- **Interpretation**: Lower is better
- **Typical range**: 0.9 - 1.1 (model matches signal variance)
- **Our results**: ~1.0 (good baseline performance)

### MSE (Mean Squared Error)
- **Definition**: Average squared prediction error
- **Interpretation**: Absolute error magnitude (power units)
- **Use case**: Shows sensitivity to outliers

### SE (Spectral Efficiency)
- **Definition**: log₂(1 + SNR_actual / SNR_predicted)
- **Units**: bits/sec/Hz
- **Interpretation**: Communication-relevant metric
- **Higher is better**: ~8-10 bits/sec/Hz is typical

---

## Common Tasks

### View Comparison Results
```bash
cat online_csi/output/fdd/metrics/comparison_summary.json
```

### Check Per-Condition Performance
```bash
grep -i "cm_A_ds_030" online_csi/output/fdd/metrics/online_test_detailed.csv | head -10
```

### Plot Performance Over Time
The plots are automatically generated and saved to:
```
online_csi/output/fdd/plots/
```

View any PNG file to see:
- Error evolution across 54 test samples
- Adaptation speed by condition
- Online vs offline comparison

### Export Results to Another Format
All metrics are saved as CSV (for spreadsheets) and JSON (for scripts):
```bash
python -c "
import pandas as pd
import json

# Read CSV
df = pd.read_csv('online_csi/output/fdd/metrics/online_test_detailed.csv')
print(df.describe())

# Read JSON
with open('online_csi/output/fdd/metrics/comparison_summary.json') as f:
    summary = json.load(f)
    print(f'NMSE Improvement: {summary[\"comparison\"][\"nmse_improvement\"]:.2f}%')
"
```

---

## Troubleshooting

### "Data not found"
```bash
# Check if data exists
find z_artifacts/data -type d -name "cm_*" | wc -l
# Should show 54 directories (27 × 2 for train/test)
```

### "CUDA out of memory"
```bash
# Use CPU instead
python online_csi/run_experiment.py --scenario FDD --device cpu

# Or reduce batch size in config.py
config.offline_batch_size = 16  # Instead of 32
```

### "Module import error"
```bash
# Ensure you're in project root
cd /home/omar/csi/OLCSI_on_CSI4CAST
python online_csi/run_experiment.py --scenario FDD
```

### "Offline model not found"
```bash
# Run offline training first
python online_csi/run_experiment.py --scenario FDD --offline-only

# Then run online training
python online_csi/run_experiment.py --scenario FDD --online-only
```

---

## Key Features

✅ **Automatic Data Discovery** - Finds all 27 channel conditions  
✅ **Unified Dataset** - Single training set with all conditions  
✅ **Per-Batch Tracking** - Logs metrics for each sample  
✅ **Error Evolution** - Colored by condition for analysis  
✅ **Full Pipeline** - Data → Offline → Online → Evaluate → Visualize  
✅ **Publication-Ready Plots** - 9 different visualization types  
✅ **CSV + JSON Export** - Both formats for analysis  
✅ **Model Checkpoints** - Saves trained weights  
✅ **Reproducible** - All parameters saved in experiment_config.json  
✅ **Scalable** - Easy to extend with new conditions or scenarios  

---

## Next Steps

1. **Run both scenarios**: `FDD` and `TDD`
2. **Compare results**: Check `comparison_summary.json`
3. **Analyze plots**: Look at error evolution by condition
4. **Customize hyperparameters**: Edit `config.py` for different settings
5. **Export metrics**: Use CSV files for further analysis

---

## References

- **Results**: See [RESULTS_SUMMARY.md](./RESULTS_SUMMARY.md)
- **Implementation**: See [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md)
- **Code Structure**: See individual module docstrings

---

## Citation

If you use this framework in research, please cite:

```bibtex
@software{olcsi2026,
  title={Online Learning Framework for CSI Prediction},
  author={Omar},
  year={2026},
  url={https://github.com/your-repo}
}
```

---

**Happy experimenting! 🚀**

For detailed information, see [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md)
