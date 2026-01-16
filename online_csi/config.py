"""Configuration management for online/offline training experiments."""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Main configuration for online vs offline comparison experiments."""
    
    # Experiment parameters
    scenario: str  # "TDD" or "FDD"
    seed: int = 42
    device: str = "cuda"  # "cuda" or "cpu"
    
    # Data paths
    data_root: Path = field(default_factory=lambda: Path("z_artifacts/data"))
    output_root: Path = field(default_factory=lambda: Path("online_csi/output"))
    
    # Antenna mode
    is_separate_antennas: bool = True  # Always True for this experiment
    
    # Model parameters
    model_name: str = "CNN_Online"
    num_subcarriers: int = 300
    hist_len: int = 16
    pred_len: int = 4
    cnn_hidden_dims: List[int] = field(default_factory=lambda: [32, 64, 128])
    mlp_hidden_dim: int = 512
    
    # Online training parameters
    online_lr: float = 1e-3
    online_warmup_batches: int = 0  # No warmup (process all samples)
    online_max_batches: Optional[int] = None  # None = use all data
    
    # Offline training parameters
    offline_lr: float = 1e-3
    offline_batch_size: int = 32
    offline_epochs: int = 100
    offline_momentum: float = 0.9
    offline_weight_decay: float = 1e-4
    
    # Loss function
    loss_fn: str = "nmse"  # "nmse", "mse", or "se"
    
    # Data discovery
    channel_models: List[str] = field(default_factory=lambda: ["A", "B", "C", "D", "E"])
    delay_spreads_ns: List[int] = field(
        default_factory=lambda: [30, 50, 100, 200, 300, 400]
    )
    speeds_kmph: List[int] = field(
        default_factory=lambda: [1, 3, 6, 9, 10, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45]
    )
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.scenario in ["TDD", "FDD"], f"Invalid scenario: {self.scenario}"
        assert self.device in ["cuda", "cpu"], f"Invalid device: {self.device}"
        assert self.is_separate_antennas is True, "Only separate antenna mode is supported"
        assert self.model_name == "CNN_Online", "Only CNN_Online model is supported"
        
    @property
    def output_dir(self) -> Path:
        """Get output directory for current scenario."""
        return self.output_root / self.scenario.lower()
    
    @property
    def metrics_dir(self) -> Path:
        """Get metrics output directory."""
        return self.output_dir / "metrics"
    
    @property
    def plots_dir(self) -> Path:
        """Get plots output directory."""
        return self.output_dir / "plots"
    
    @property
    def models_dir(self) -> Path:
        """Get models checkpoint directory."""
        return self.output_dir / "models"
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory."""
        return self.output_dir / "logs"
    
    def create_output_dirs(self):
        """Create all output directories."""
        for d in [self.metrics_dir, self.plots_dir, self.models_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)
