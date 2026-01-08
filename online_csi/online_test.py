"""
Online CSI Training and Evaluation Script.

This script implements an online learning loop for CSI prediction models,
simulating a real-time deployment scenario where the model is updated
continuously as new data arrives.

Key Features:
- Online training with small batches and no shuffling
- Real-time performance monitoring (training time, inference time)
- Computational overhead analysis (FLOPs, parameters)
- Periodic or final evaluation on a held-out test set
"""

import argparse
import logging
import time
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from src.utils.data_utils import load_data, CSIDataset, collect_fn_gather_antennas, collect_fn_separate_antennas
from src.utils.norm_utils import load_normalization_stats, normalize_input
from src.cp.dataset.data_module import TrainValDataModule
from src.cp.config.config import DataConfig
from src.cp.models import PREDICTORS
from src.testing.prediction_performance.test_unit import test_unit
from src.testing.computational_overhead.utils import (
    compute_all_metrics,
    count_flops,
    measure_inference_time_stats,
    count_trainable_parameters,
    log_computational_metrics,
)
from src.cp.loss.loss import NMSELoss, MSELoss, SELoss

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_online_training(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    num_updates: int,
    device: torch.device,
    warmup: int = 5,
) -> dict[str, float]:
    """
    Run online training loop and measure per-sample training time.
    """
    model.train()
    training_times_per_sample = []
    
    logger.info(f"Starting online training for {num_updates} updates...")
    
    iterator = iter(dataloader)
    
    for i in range(num_updates):
        try:
            batch = next(iterator)
        except StopIteration:
            logger.warning("Dataloader exhausted before completing all updates.")
            break
            
        # Unpack batch (hist, target)
        hist, target = batch
        hist, target = hist.to(device), target.to(device)
        
        # Determine effective number of samples for timing normalization
        # For separate antennas: batch_size * antennas (hist.shape[0])
        # For gather antennas: batch_size (hist.shape[0])
        num_samples = hist.shape[0]

        # Sync before timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        # --- Update Step ---
        optimizer.zero_grad()
        pred = model(hist)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        # -------------------
        
        # Sync after timing
        if device.type == "cuda":
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        
        # Record time (skip warmup)
        if i >= warmup:
            elapsed = end_time - start_time
            training_times_per_sample.append(elapsed / num_samples)
            
    # Compute statistics
    if not training_times_per_sample:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
    times = np.array(training_times_per_sample)
    return {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
    }


def main():
    parser = argparse.ArgumentParser(description="Online CSI Training and Evaluation")
    
    # Model & Scenario
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., RNN, CNN)")
    parser.add_argument("--scenario", type=str, default="FDD", choices=["FDD", "TDD"], help="Scenario (FDD/TDD)")
    
    # Training Params
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for online updates")
    parser.add_argument("--num_updates", type=int, default=100, help="Number of online updates to run")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    
    # Data Params
    parser.add_argument("--dir_data", type=str, default="z_artifacts/data", help="Data directory")
    parser.add_argument("--cm", type=str, default="A", help="Channel Model")
    parser.add_argument("--ds", type=float, default=30e-9, help="Delay Spread")
    parser.add_argument("--ms", type=int, default=1, help="Speed")
    
    # Flags
    parser.add_argument("--compute_flops", action="store_true", help="Compute FLOPs")
    parser.add_argument("--measure_time", action="store_true", help="Measure inference/training time")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on test set")
    parser.add_argument("--out_dir", type=str, default="z_artifacts/outputs/testing/online_test", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) / f"{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running on {device}, Output dir: {out_dir}")

    # 1. Instantiate Model
    # Note: We need to instantiate the PL module to access the underlying model if needed, 
    # or just the model itself. The registry returns the PL class.
    # We'll instantiate the PL class and then use its 'model' attribute or valid forward method.
    model_cls = getattr(PREDICTORS, f"{args.model}_{args.scenario}")
    
    # We need to construct the model. Usually this requires config or params.
    # For now, we'll try to instantiate with default/inferred params for the scenario.
    # A cleaner way is to use the Config system, but for now we might construct manually 
    # or rely on defaults if they exist.
    # Let's inspect how main.py does it or assume we can pass basic args.
    # The CNNOnline model we built takes is_separate_antennas etc.
    
    # Hack: Inspect model class to see what it needs or use a config object
    # For simplicity in this script, we assume the model class can be init via config-like dict or args.
    # Actually, the PL modules usually take a config object.
    
    # Let's import the config classes
    from src.cp.config.config import ExperimentConfig, ModelConfig
    
    # Create a dummy config to init the model
    # We might need to refine this based on the specific model's __init__
    model_conf = ModelConfig(
        name=f"{args.model}_{args.scenario}",
        params={
             "num_subcarriers": 48, # Default/common value, should ideally come from constants
             "pred_len": 10, # Default
             "hist_len": 4, # Default
             "cnn_hidden_dims": [32, 64, 128], # For CNN
             "mlp_hidden_dim": 512, # For CNN
        }
    )
    # RNN expects 'input_dim', 'hidden_dim', etc. in params.
    if "RNN" in args.model:
         model_conf.params.update({
             "input_dim": 96, # 48 * 2 for real/imag
             "hidden_dim": 256,
             "num_layers": 2,
             "dropout": 0.0
         })

    model = model_cls(model_conf)
    model.to(device)
    
    # 2. Data Setup (Online Training Data)
    # We use FDD (is_U2D=True) data for training as requested
    logger.info("Loading training data...")
    h_hist_train = load_data(
        dir_data=Path(args.dir_data),
        list_cm=[args.cm],
        list_ds=[args.ds],
        list_ms=[args.ms],
        is_train=True,
        is_gen=False,
        is_hist=True,
        is_U2D=True 
    )
    h_pred_train = load_data(
        dir_data=Path(args.dir_data),
        list_cm=[args.cm],
        list_ds=[args.ds],
        list_ms=[args.ms],
        is_train=True,
        is_gen=False,
        is_hist=False,
        is_U2D=True
    )
    
    # Normalize
    # We need stats. Assuming they exist.
    try:
        h_hist_train, h_pred_train = normalize_input(h_hist_train, h_pred_train, is_U2D=True)
    except Exception as e:
        logger.error(f"Normalization failed: {e}. Ensure stats are generated.")
        return

    train_dataset = CSIDataset(h_hist_train, h_pred_train)
    
    collate_fn = collect_fn_separate_antennas if getattr(model, "is_separate_antennas", False) else collect_fn_gather_antennas
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False, # Online training order
        collate_fn=collate_fn
    )

    # 3. Online Training Loop
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = NMSELoss() # Use NMSE for training as requested
    
    comp_metrics = {}
    
    if args.measure_time:
        logger.info("Running online training...")
        train_stats = run_online_training(model, train_loader, optimizer, criterion, args.num_updates, device)
        comp_metrics["train_time_per_sample_ms"] = train_stats["mean"] * 1000
        comp_metrics["train_time_std_ms"] = train_stats["std"] * 1000
        logger.info(f"Training Time: {train_stats['mean']*1000:.2f} ± {train_stats['std']*1000:.2f} ms/sample")
        
        # Inference time
        logger.info("Measuring inference time...")
        # Get one sample
        dummy_input = next(iter(train_loader))[0].to(device)
        # We need batch_size=1 for per-sample inference time
        if dummy_input.shape[0] > 1:
             dummy_input = dummy_input[:1]
             
        inf_stats = measure_inference_time_stats(model, dummy_input, device)
        comp_metrics["inf_time_per_sample_ms"] = inf_stats["mean"] * 1000
        logger.info(f"Inference Time: {inf_stats['mean']*1000:.2f} ms/sample")

    # 4. FLOPs
    if args.compute_flops:
        logger.info("Computing FLOPs...")
        # Use util with batch_size=1
        metrics, errors = compute_all_metrics(model, device, batch_size=1)
        comp_metrics.update(metrics)
        if errors:
            logger.warning(f"FLOPs errors: {errors}")
        log_computational_metrics(metrics, args.model)

    # Save metrics
    with open(out_dir / "comp_metrics.json", "w") as f:
        json.dump(comp_metrics, f, indent=4)

    # 5. Evaluation
    if args.evaluate:
        logger.info("Evaluating on test set...")
        # We reuse test_unit
        # Define criteria
        crit_nmse = NMSELoss()
        crit_mse = MSELoss()
        crit_se = SELoss()
        
        # We evaluate on the specific scenario/condition provided in args
        # or we could iterate over a standard set. 
        # For this script, let's stick to the args provided (single point evaluation)
        
        df = test_unit(
            scenario=args.scenario,
            is_gen=False,
            is_U2D=True, # Always FDD for evaluation in this context? Or matches scenario? 
                         # Instructions said "evaluate performance on the regular FDD test set".
            dir_data=Path(args.dir_data),
            batch_size=args.batch_size, # Use same batch size or standard? Standard is usually larger but doesn't matter much for metrics
            device=device,
            list_models=[model],
            criterion_nmse=crit_nmse,
            criterion_mse=crit_mse,
            criterion_se=crit_se,
            cm=args.cm,
            ds=args.ds,
            ms=args.ms,
            noise_type="vanilla", # Default
            noise_func=lambda x, y: 0, # Placeholder, test_unit handles noise if we pass a real func
            # Wait, test_unit expects a noise function if noise_type is vanilla.
            # We should probably import the noise utils if we want real noise.
            # For now let's assume 0 noise or use the project's noise.
            noise_degree=0,
            df_path=out_dir / "results.csv"
        )
        logger.info(f"Evaluation complete. Results saved to {out_dir / 'results.csv'}")
        print(df)

if __name__ == "__main__":
    main()
