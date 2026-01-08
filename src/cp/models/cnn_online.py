import math
import logging
import torch
import torch.nn as nn

from src.cp.config.config import ExperimentConfig
from src.cp.models.common.base import BaseCSIModel
from src.utils.data_utils import HIST_LEN, PRED_LEN, NUM_SUBCARRIERS

logger = logging.getLogger(__name__)


class CNNOnlineModel(nn.Module):
    """CNN model for online CSI prediction (Core Module).

    Architecture:
        - Conv2D -> BN -> MaxPool
        - Conv2D -> MaxPool
        - Conv2D -> MaxPool -> Flatten
        - MLP Decoders
    """

    def __init__(
        self,
        num_subcarriers=NUM_SUBCARRIERS,
        hist_len=HIST_LEN,
        pred_len=PRED_LEN,
        cnn_hidden_dims=[32, 64, 128],
        mlp_hidden_dim=512,
        **kwargs,
    ):
        super().__init__()

        self.subcarriers = num_subcarriers
        self.hist_len = hist_len
        self.pred_len = pred_len

        # Input channels: 1 (treated as grayscale image [H, W])
        in_channels = 1

        # Calculate resulting dimensions after pooling
        curr_h = self.hist_len
        curr_w = self.subcarriers * 2  # Real + Imag

        # 1st Block: Conv2D(32, kernel=3) -> BatchNorm -> MaxPool2D
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, cnn_hidden_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_hidden_dims[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        curr_h = math.ceil(curr_h / 2)
        curr_w = math.ceil(curr_w / 2)

        # 2nd Block: Conv2D(64, kernel=3) -> MaxPool2D
        self.conv2 = nn.Sequential(
            nn.Conv2d(cnn_hidden_dims[0], cnn_hidden_dims[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        curr_h = math.ceil(curr_h / 2)
        curr_w = math.ceil(curr_w / 2)

        # 3rd Block: Conv2D(128, kernel=3) -> MaxPool2D -> Flatten
        self.conv3 = nn.Sequential(
            nn.Conv2d(cnn_hidden_dims[1], cnn_hidden_dims[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        curr_h = math.ceil(curr_h / 2)
        curr_w = math.ceil(curr_w / 2)

        self.flatten_dim = cnn_hidden_dims[2] * curr_h * curr_w
        logger.info(
            f"CNN output shape: [{cnn_hidden_dims[2]}, {curr_h}, {curr_w}] -> Flatten: {self.flatten_dim}"
        )

        # Two-stage MLP decoder
        # MLP width constructor: reshape to [batch, 1, width=subcarriers*2]
        self.width_constructor = nn.Sequential(
            nn.Flatten(), nn.Linear(self.flatten_dim, self.subcarriers * 2), nn.ReLU()
        )

        # MLP height constructor: reshape to final [batch, pred_len, subcarriers*2]
        self.height_constructor = nn.Sequential(
            nn.Linear(self.subcarriers * 2, self.pred_len * self.subcarriers * 2)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor.
               If is_separate_antennas=True: [Batch, Hist_Len, Subcarriers*2] (Real)
               If is_separate_antennas=False: [Batch, Antennas, Hist_Len, Subcarriers] (Complex)

        Returns:
            out: Predicted CSI. [Batch, Pred_Len, Subcarriers*2] or [Batch, Antennas, Pred_Len, Subcarriers*2]
        """

        # Handle input shapes and types
        is_complex_input = x.is_complex()
        original_shape = x.shape

        if is_complex_input:
            # Assume [Batch, Antennas, Hist_Len, Subcarriers]
            # Convert to [Batch*Antennas, Hist_Len, Subcarriers*2]
            # view_as_real adds a last dim of size 2.
            # x: [B, A, T, F] -> [B, A, T, F, 2] -> [B*A, T, F*2]
            x_real = torch.view_as_real(x)
            B, A, T, F, _ = x_real.shape
            x = x_real.view(B * A, T, F * 2)
        else:
            # Real input
            if x.dim() == 4:
                # Assume [Batch, Antennas, Hist_Len, Features]
                B, A, T, F = x.shape
                x = x.view(B * A, T, F)
            elif x.dim() == 3:
                # Standard [Batch, Hist_Len, Features]
                pass
            else:
                raise ValueError(f"Unexpected input dimension: {x.dim()}")

        # Ensure 4D for Conv2D: [Batch_Effective, Channel=1, H, W]
        # x is [Batch_Eff, Hist_Len, Feat]
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Stage 1: Width constructor
        x_width = self.width_constructor(x)

        # Stage 2: Height constructor
        x_height = self.height_constructor(x_width)

        # Final reshape
        B_eff = x.shape[0]
        out = x_height.view(B_eff, self.pred_len, self.subcarriers * 2)

        # Restore dimensions if needed
        if is_complex_input or len(original_shape) == 4:
            # Input was [B, A, ...]
            # Out is [B*A, Pred, F*2] -> [B, A, Pred, F*2]
            B = original_shape[0]
            A = original_shape[1]
            out = out.view(B, A, self.pred_len, -1)

        return out


class CNNOnline(BaseCSIModel):
    """PyTorch Lightning wrapper for CNNOnline model.
    
    Architecture (as per GEMINI.md):
        - Conv2D(32, kernel=3) -> BatchNorm -> MaxPool2D
        - Conv2D(64, kernel=3) -> MaxPool2D
        - Conv2D(128, kernel=3) -> MaxPool2D -> Flatten
        - Two-stage MLP decoder
    """

    def __init__(self, config: ExperimentConfig, **kwargs):
        super().__init__(
            optimizer_config=config.optimizer,
            scheduler_config=config.scheduler,
            loss_config=config.loss,
        )
        self.name = "CNN_Online"
        self.is_separate_antennas = config.model.is_separate_antennas
        self.save_hyperparameters({"model": config.model})

        # Initialize the core model
        self.model = CNNOnlineModel(**config.model.params)

    def forward(self, x):
        return self.model(x)