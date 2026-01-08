import argparse
import logging
import time
import json
import csv
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import lightning.pytorch as pl
from tqdm import tqdm

from src.cp.config.config import ExperimentConfig
from src.cp.models import PREDICTORS
from src.cp.dataset.data_module import TrainValDataModule
from src.cp.loss.loss import MSELoss, NMSELoss, SELoss
from src.testing.prediction_performance.test_unit import test_unit
from src.testing.computational_overhead.utils import (
    measure_model_time,
    count_flops,
    get_input_data_for_model,
    count_total_parameters,
    count_trainable_parameters
)
from src.utils.data_utils import HIST_LEN, NUM_SUBCARRIERS, TOT_ANTENNAS, PRED_LEN
from src.utils.dirs import DIR_DATA, DIR_OUTPUTS
from src.utils.time_utils import get_current_time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_online_training(model, datamodule, optimizer, num_updates, device, warmup=5):
    """
    Run online training updates and measure full update time.
    Returns list of per-sample full-update times (seconds).
    """
    model.train()
    model.to(device)
    
    # Get dataloader
    train_loader = datamodule.train_dataloader()
    iterator = iter(train_loader)
    
    times = []
    
    pbar = tqdm(range(num_updates), desc="Online Training")
    for i in pbar:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
            
        hist, target = batch
        hist = hist.to(device)
        target = target.to(device)
        
        num_samples = hist.shape[0]
        
        # Start timing full update
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        optimizer.zero_grad()
        loss = model.training_step((hist, target), i)
        loss.backward()
        optimizer.step()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        
        if i >= warmup:
            per_sample_time = elapsed / num_samples
            times.append(per_sample_time)
            
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return times

def main():
    parser = argparse.ArgumentParser(description="Online CSI Training and Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Model name (RNN, CNN)")
    parser.add_argument("--scenario", type=str, default="FDD", choices=["FDD", "TDD"], help="Scenario")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for online updates")
    parser.add_argument("--num_updates", type=int, default=100, help="Number of online updates")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu, cuda)")
    parser.add_argument("--gather_antennas", action="store_true", help="Use gather antennas mode")
    parser.add_argument("--compute_flops", action="store_true", help="Compute FLOPs")
    parser.add_argument("--measure_time", action="store_true", help="Measure training/inference time")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on test set after training")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    is_separate_antennas = not args.gather_antennas
    scenario = args.scenario
    is_U2D = (scenario == "FDD")
    
    config = ExperimentConfig()
    config.data.batch_size = args.batch_size
    config.data.is_U2D = True # Online training typically uses FDD/U2D
    config.data.shuffle = False # Streaming
    config.data.is_separate_antennas = is_separate_antennas
    
    config.model.name = f"{args.model}_{scenario}"
    config.model.is_separate_antennas = is_separate_antennas
    
    if args.model == "RNN":
        config.model.params = {
            "dim_data": NUM_SUBCARRIERS * 2,
            "rnn_hidden_dim": NUM_SUBCARRIERS * 4,
            "rnn_num_layers": 4,
            "pred_len": PRED_LEN,
        }
    elif args.model == "CNN":
        config.model.params = {
            "num_subcarriers": NUM_SUBCARRIERS,
            "pred_len": PRED_LEN,
            "hist_len": HIST_LEN,
            "cnn_hidden_dims": [32, 64, 128],
            "mlp_hidden_dim": 512,
        }

    model_class = getattr(PREDICTORS, config.model.name)
    logger.info(f"Initializing model: {config.model.name}")
    
    if args.model == "RNN":
        model = model_class(config)
    else:
        model = model_class(
            optimizer_config=config.optimizer,
            scheduler_config=config.scheduler,
            loss_config=config.loss,
            params=config.model.params
        )
    model.to(device)
    
    dm = TrainValDataModule(config.data)
    dm.setup()
    
    # Optimizer (Adam, no weight decay for initial online tests)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.params["lr"], weight_decay=0.0)
    
    timestamp = get_current_time()
    if args.outdir:
        out_path = Path(args.outdir)
    else:
        out_path = Path(DIR_OUTPUTS) / "testing" / "online_test" / timestamp
    out_path.mkdir(parents=True, exist_ok=True)
    
    comp_res = {}
    
    # === 1. Online Training ===
    training_time_avg = np.nan
    training_time_std = np.nan
    
    if args.measure_time:
        logger.info(f"Starting online training for {args.num_updates} updates...")
        training_times = run_online_training(model, dm, optimizer, args.num_updates, device)
        if training_times:
            training_time_avg = np.mean(training_times)
            training_time_std = np.std(training_times)
            logger.info(f"Full-update training time per sample: {training_time_avg*1000:.2f} ms")

    # === 2. Computational Metrics (Standard Format) ===
    if args.compute_flops or args.measure_time:
        logger.info("Computing computational metrics...")
        input_data = get_input_data_for_model(model, 1, device)
        
        total_params = count_total_parameters(model)
        trainable_params = count_trainable_parameters(model)
        flops, flops_error = count_flops(model, input_data) if args.compute_flops else (np.nan, None)
        
        list_inf_time = measure_model_time(
            model=model, mode="inference", device=device, input_data=input_data, desc="Inference timing"
        )
        inference_time_avg = np.mean(list_inf_time)
        inference_time_std = np.std(list_inf_time)
        
        # Create standard report row
        df_comp = pd.DataFrame({
            "Scenario": [scenario],
            "Model": [args.model],
            "Batch_Size": [1],
            "Total_Params": [total_params],
            "Trainable_Params": [trainable_params],
            "Total_Params_M": [total_params / 1e6],
            "Trainable_Params_M": [trainable_params / 1e6],
            "FLOPS": [flops],
            "MFLOPS": [flops / 1e6 if not np.isnan(flops) else np.nan],
            "GFLOPS": [flops / 1e9 if not np.isnan(flops) else np.nan],
            "Inference_Time_Avg_ms": [inference_time_avg * 1000],
            "Inference_Time_Std_ms": [inference_time_std * 1000],
            "Training_Time_Avg_ms": [training_time_avg * 1000],
            "Training_Time_Std_ms": [training_time_std * 1000],
        })
        
        comp_csv = out_path / "computational_overhead.csv"
        df_comp.to_csv(comp_csv, index=False)
        logger.info(f"Saved computational metrics to {comp_csv}")
        
        with open(out_path / "computational_overhead.json", "w") as f:
            json.dump(df_comp.to_dict(orient="records")[0], f, indent=4)

    # === 3. Evaluation (Standard Format) ===
    if args.evaluate:
        logger.info("Evaluating on FDD test set...")
        result_csv = out_path / "result.csv"
        
        from src.utils.data_utils import LIST_CHANNEL_MODEL, LIST_DELAY_SPREAD, LIST_MIN_SPEED_TEST
        
        criterion_nmse = NMSELoss().to(device)
        criterion_mse = MSELoss().to(device)
        criterion_se = SELoss(SNR=10).to(device)
        
        for cm in LIST_CHANNEL_MODEL:
            for ds in LIST_DELAY_SPREAD:
                for ms in LIST_MIN_SPEED_TEST:
                    logger.info(f"Testing combination: CM={cm}, DS={ds}, MS={ms}")
                    test_unit(
                        scenario="FDD",
                        is_gen=False,
                        is_U2D=True,
                        dir_data=Path(DIR_DATA),
                        batch_size=args.batch_size,
                        device=device,
                        list_models=[model],
                        criterion_nmse=criterion_nmse,
                        criterion_mse=criterion_mse,
                        criterion_se=criterion_se,
                        cm=cm,
                        ds=ds,
                        ms=ms,
                        noise_type="vanilla",
                        noise_func=lambda x, y: 0, # Clean evaluation
                        noise_degree=0,
                        df_path=result_csv
                    )
        logger.info(f"Evaluation complete. Results saved to {result_csv}")

        # === 4. Performance Summary (Expected format) ===
        try:
            from src.testing.results.gather_results import parse_array_string
            from tabulate import tabulate
            
            df_perf = pd.read_csv(result_csv)
            # Flatten the step-wise results
            df_perf = parse_array_string(df_perf, ["nmse_mean", "se_mean"])
            
            # Aggregate across all scenarios and steps for a high-level summary
            summary_stats = df_perf[["nmse_mean", "se_mean"]].mean().to_frame().T
            summary_stats["Model"] = args.model
            summary_stats["Scenario"] = scenario
            
            cols = ["Model", "Scenario", "nmse_mean", "se_mean"]
            print("\n📈 Prediction Performance Summary (Averaged across all FDD test scenarios):")
            print(tabulate(summary_stats[cols].values, headers=cols, tablefmt="psql", floatfmt=".4f"))
            
        except Exception as e:
            logger.warning(f"Could not generate performance summary: {e}")

if __name__ == "__main__":
    main()