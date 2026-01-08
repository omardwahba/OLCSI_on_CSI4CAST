"""Small data inspection utility for Phase One (data dimensions & ordering).

Usage examples:
    python -m src.utils.inspect_data --cm A --ds 30e-9 --ms 1 --is_U2D True --n_samples 10

This script reuses existing project utilities and datamodule to:
- Load a subset's raw tensors (`_load_data`)
- Print raw and collated batch shapes
- Inspect whether the training dataloader will shuffle
- Check for time-overlap between consecutive samples (heuristic)
- Verify normalization stats are available and that normalization preserves shapes

"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    import torch
except Exception as e:
    print("This script requires PyTorch (torch). Please activate the project's conda/mamba environment and install dependencies. Error:", e)
    raise

from src.cp.config.config import DataConfig
from src.cp.dataset.data_module import TrainValDataModule
from src.utils.data_utils import _load_data, load_data
from src.utils.norm_utils import load_normalization_stats, normalize_input
from src.utils.dirs import DIR_DATA


def print_banner(msg: str):
    print("\n" + "=" * 60)
    print(msg)
    print("=" * 60 + "\n")


def check_overlap(H_hist: torch.Tensor, n_checks: int = 5) -> bool:
    """Heuristic check for sliding-window overlap between consecutive samples.

    For consecutive sample pairs (i, i+1) we test if the last k timesteps of
    sample i equal the first k timesteps of sample i+1 for any k in 1..(hist_len-1).
    If such equality is found for any inspected pair and k, we flag "overlap found".

    Returns True if overlap *likely* exists, False otherwise.
    """
    if len(H_hist) < 2:
        return False

    hist_len = H_hist.shape[2]
    num_pairs = min(len(H_hist) - 1, n_checks)
    print(H_hist[0,0,:,:10])
    print("--------------------------------")
    print(H_hist[1,0,:,:10])
    return False
    for i in range(num_pairs):
        a = H_hist[i]
        b = H_hist[i + 1]
        # a shape: [antennas, hist_len, subcarriers]
        for k in range(1, hist_len):
            a_tail = a[0, -k:, :]
            b_head = b[0, :k, :]
            if torch.equal(a_tail, b_head):
                print(f"Overlap detected between sample {i} and {i+1} with k={k}")
                return True
    return False


def inspect_subset(dir_data: Path, cm: str, ds: float, ms: int, is_train: bool, is_U2D: bool, n_samples: int = 10):
    print_banner("Subset raw file inspection")
    try:
        H_hist = _load_data(dir_data=dir_data, cm=cm, ds=ds, ms=ms, is_train=is_train, is_gen=False, is_hist=True, is_U2D=is_U2D, num_load=n_samples)
        H_pred = _load_data(dir_data=dir_data, cm=cm, ds=ds, ms=ms, is_train=is_train, is_gen=False, is_hist=False, is_U2D=is_U2D, num_load=n_samples)
    except FileNotFoundError as e:
        print(f"ERROR: Could not find subset files: {e}")
        return None

    print(f"Loaded subset: cm={cm}, ds={ds}, ms={ms}, is_train={is_train}, is_U2D={is_U2D}")
    print(f"  Raw shapes: H_hist={tuple(H_hist.shape)} (dtype={H_hist.dtype}), H_pred={tuple(H_pred.shape)} (dtype={H_pred.dtype})")
    print(f"  Example sample 0 shapes: hist={tuple(H_hist[0].shape)}, pred={tuple(H_pred[0].shape)}")

    # Overlap check
    overlap = check_overlap(H_hist, n_checks=min(10, len(H_hist) - 1))
    print(f"  Overlap heuristic: {'LIKELY (found)' if overlap else 'NOT found (likely independent samples)'}")

    # Normalization stats availability
    try:
        stats = load_normalization_stats(dir_data=Path(dir_data), is_U2D=is_U2D)
        print(f"  Normalization stats loaded (mean_r={stats.mean_r.item():.6f}, std_r={stats.std_r.item():.6f})")

        # Verify normalization preserves shapes
        H_hist_norm, H_pred_norm = normalize_input(H_hist, H_pred, is_U2D=is_U2D)
        print(f"  After normalization shapes: H_hist={tuple(H_hist_norm.shape)}, H_pred={tuple(H_pred_norm.shape)}")
    except FileNotFoundError:
        print("  Normalization stats NOT found. Run `python -m src.utils.norm_utils` to compute them.")

    return (H_hist, H_pred)


def inspect_datamodule(dir_data: Path, is_U2D: bool, batch_size: int = 4, shuffle: bool = True, is_separate_antennas: bool = True):
    print_banner("DataModule & Dataloader inspection")

    data_cfg = DataConfig(dir_dataset=str(dir_data), batch_size=batch_size, is_U2D=is_U2D, shuffle=shuffle, is_separate_antennas=is_separate_antennas)
    dm = TrainValDataModule(data_cfg)

    # Setup will perform stratified split, normalization and add noise
    dm.setup()

    # dataset sizes
    train_len = len(dm.train_dataset)
    val_len = len(dm.val_dataset)
    print(f"  Train dataset size: {train_len}")
    print(f"  Val dataset size:   {val_len}")

    # get shapes via helper
    hist_shape, pred_shape = dm.get_data_shapes()
    print(f"  Collated batch shapes from dataloader: hist={hist_shape}, pred={pred_shape}")
    print(f"  (Data type) Collated batch shapes from dataloader: hist={hist_shape}, pred={pred_shape}")
    # show dataloader config (shuffle flag)
    loader = dm.train_dataloader()
    print(f"  DataLoader shuffle flag (will shuffle each epoch): {data_cfg.shuffle}")

    # show a single batch content summary
    batch_hist, batch_pred = next(iter(loader))
    print(f"  Example batch tensors shapes: hist={tuple(batch_hist.shape)}, pred={tuple(batch_pred.shape)}")

    return dm


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cm", type=str, default="A", help="Channel model (A/C/D)")
    parser.add_argument("--ds", type=float, default=30e-9, help="Delay spread (seconds), e.g., 30e-9")
    parser.add_argument("--ms", type=int, default=1, help="Min speed (m/s) e.g., 1")
    parser.add_argument("--is_train", action="store_true", help="Load training split (default behavior is test files if omitted)")
    parser.add_argument("--is_U2D", action="store_true", help="Use FDD (U2D) files (prediction will use H_D_pred.pt)")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples to load from subset for raw inspection")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size to test datamodule dataloader")
    parser.add_argument("--no_shuffle", dest="shuffle", action="store_false", help="Disable dataloader shuffling (for deterministic behavior)")
    parser.add_argument("--separate_antennas", dest="is_separate_antennas", action="store_true", help="Use separate-antennas collation (default True)")
    parser.add_argument("--gather_antennas", dest="is_separate_antennas", action="store_false", help="Use gather-antennas collation (complex-valued)")

    args = parser.parse_args(argv)

    dir_data = Path(DIR_DATA)
    print(f"Inspecting data directory: {dir_data}")

    # Subset inspection
    subset_res = inspect_subset(dir_data=dir_data, cm=args.cm, ds=args.ds, ms=args.ms, is_train=args.is_train, is_U2D=args.is_U2D, n_samples=args.n_samples)

    # DataModule inspection
    dm = inspect_datamodule(dir_data=dir_data, is_U2D=args.is_U2D, batch_size=args.batch_size, shuffle=args.shuffle, is_separate_antennas=args.is_separate_antennas)

    print_banner("Done — Phase One checks completed")


if __name__ == "__main__":
    main(sys.argv[1:])
