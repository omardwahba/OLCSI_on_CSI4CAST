import unittest
import torch
import shutil
from pathlib import Path
from src.cp.models.cnn_online import CNNOnline, CNNOnlineModel
from src.cp.config.config import ExperimentConfig
from src.utils.data_utils import HIST_LEN, NUM_SUBCARRIERS, PRED_LEN

class TestCNNOnline(unittest.TestCase):
    def setUp(self):
        # Create a default configuration
        self.config = ExperimentConfig()
        
        # Configure for CNN
        self.config.model.name = "CNN_TDD"
        self.config.model.is_separate_antennas = True
        self.config.model.params = {
            "num_subcarriers": NUM_SUBCARRIERS,
            "pred_len": PRED_LEN,
            "hist_len": HIST_LEN,
            "cnn_hidden_dims": [32, 64, 128],
            "mlp_hidden_dim": 512,
        }
        
        # Initialize model with Config
        self.model = CNNOnline(self.config)

    def test_forward_pass_separate_antennas(self):
        # Shape: [Batch, Hist_Len, Subcarriers*2]
        batch_size = 4
        # Assuming is_separate_antennas=True, we have real valued input
        input_tensor = torch.randn(batch_size, HIST_LEN, NUM_SUBCARRIERS * 2)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Check output shape: [Batch, Pred_Len, Subcarriers*2]
        expected_shape = (batch_size, PRED_LEN, NUM_SUBCARRIERS * 2)
        self.assertEqual(output.shape, expected_shape, f"Expected shape {expected_shape}, got {output.shape}")

    def test_forward_pass_combined_antennas(self):
        # Test the core model handling of complex 4D input
        # Shape: [Batch, Antennas, Hist_Len, Subcarriers] (Complex)
        batch_size = 2
        antennas = 4
        
        # Create complex input
        input_tensor = torch.randn(batch_size, antennas, HIST_LEN, NUM_SUBCARRIERS, dtype=torch.cfloat)
        
        # Forward pass
        # The model should treat this by flattening B*A
        output = self.model(input_tensor)
        
        # Output shape should be [Batch, Antennas, Pred_Len, Subcarriers*2]
        # (Since we return real representation of the prediction)
        expected_shape = (batch_size, antennas, PRED_LEN, NUM_SUBCARRIERS * 2)
        self.assertEqual(output.shape, expected_shape, f"Expected shape {expected_shape}, got {output.shape}")

    def test_cnn_online_model_direct(self):
        # Test the core module directly
        core_model = CNNOnlineModel(**self.config.model.params)
        batch_size = 2
        input_tensor = torch.randn(batch_size, HIST_LEN, NUM_SUBCARRIERS * 2)
        output = core_model(input_tensor)
        self.assertEqual(output.shape, (batch_size, PRED_LEN, NUM_SUBCARRIERS * 2))

    def test_model_complexity(self):
        # Print parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        # print(f"Total parameters: {total_params}")
        self.assertTrue(total_params > 0)

if __name__ == "__main__":
    unittest.main()