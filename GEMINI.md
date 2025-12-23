# Project Overview

This project, "CSI-4CAST," is a comprehensive framework for generating and evaluating Channel State Information (CSI) prediction models. It is a research project that uses 3GPP TR 38.901 channel models to generate large-scale datasets, train prediction models, and perform comprehensive evaluations. The framework is designed for both local execution and high-performance computing (HPC) environments.

The project is structured into several core modules:
-   **Data Generation (`src/data`)**: Generates realistic CSI datasets using the Sionna library, based on 3GPP TR 38.901 channel models.
-   **Channel Prediction (`src/cp`)**: A PyTorch Lightning-based module for training CSI prediction models.
-   **Noise (`src/noise`)**: Handles noise modeling and parameter calibration for robustness testing.
-   **Testing (`src/testing`)**: A comprehensive evaluation framework for assessing model performance across various dimensions, including prediction accuracy and computational overhead.

The main technologies used are Python, PyTorch, PyTorch Lightning, Sionna, and NumPy. The project is well-documented and includes sample outputs and configurations.

# Building and Running

## Environment Setup

The project uses a Conda environment. To set it up, run:
```bash
module load mamba/[mamba_version]
mamba env create -f env.yml
mamba activate csi-4cast-env
```

## Data Generation

Data generation is handled by `src/data/generator.py`. It can be run directly or using the provided SLURM script template.

**To generate training data:**
```bash
python3 -m src.data.generator --is_train
```

**To generate regular test data:**
```bash
python3 -m src.data.generator
```

**To generate generalization test data:**
```bash
python3 -m src.data.generator --is_gen
```
Debug mode is available for generating smaller datasets for testing purposes. After generating the data, it's necessary to compute normalization statistics:
```bash
python3 -m src.utils.norm_utils
```

## Model Training

Model training is handled by `src/cp/main.py`. It uses configuration files to define the training parameters.

**First, generate a configuration file:**
```bash
python3 -m src.cp.config.config --model [model_name] --output-dir [output_dir] --is_U2D [True/False] --config-file [yaml/json]
```

**Then, start the training:**
```bash
python3 -m src.cp.main --hparams_csi_pred [config_file]
```
SLURM script templates are available for running training on an HPC cluster.

## Model Testing

The testing module (`src/testing`) is used to evaluate the performance of the trained models.

**To run computational overhead testing:**
```bash
python3 -m src.testing.computational_overhead.main
```

**To run prediction performance testing:**
```bash
python3 -m src.testing.prediction_performance.main --model [model_name]
```
For large-scale testing on an HPC cluster, use the provided SLURM script templates.

## Results Processing

After running the tests, the results can be processed and analyzed:

```bash
python3 -m src.testing.results.main
```

This will check for completion of the tests, gather the results, and perform analysis.

## Visualization

The framework also includes a visualization module to generate plots and tables from the results:

```bash
python3 -m src.testing.vis.main
```

# Development Conventions

-   **Linting**: The project uses `ruff` for linting and formatting. The configuration is in `pyproject.toml`.
-   **Coding Style**: The code follows PEP 8 guidelines with a line length of 120 characters. Docstrings are used to document modules, classes, and functions.
-   **Testing**: The project has a comprehensive testing framework. New models should be added to `src/cp/models` and registered in `src/cp/models/__init__.py`. Testing configurations are in `src/testing/config.py`.
-   **Configuration**: The project uses a hierarchical configuration system based on `dataclasses`. Configurations for training are managed in `src/cp/config/config.py`.
-   **Dependencies**: Project dependencies are managed in `env.yml` for the Conda environment.
