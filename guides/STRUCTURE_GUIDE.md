# Machine Learning Project Structure Guide

This guide details the expected directory structure and specific usage rules for files within the repository. Following these rules ensures the pipeline runs correctly and components remain modular.

## Core Flow: `main.py`
**The `main.py` file is the master orchestrator for the entire analysis.**
- It should serve as the single entry point.
- It sequentially calls data loading/preprocessing, model building, training, evaluation, and plotting.
- Use imported functions from `src/` modules in `main.py` rather than putting implementation logic inside `main.py` itself.

## Configuration: `config.yaml`
- Use this file as the central source of truth for global variables, file paths, and model hyperparameters. 
- Scripts under `src/` or `scripts/` should read from `config.yaml` instead of hardcoding numeric thresholds or file paths.

## Standalone Scripts (`scripts/`)
Files in the `scripts/` directory perform specific, isolated tasks that don't depend on the main training flow.
- **Example: `scripts/preprocessing.py`**
  - This script acts as a standalone data processor. 
  - Its objective is to consume raw datasets and generate processed formats (e.g. converting `events.parquet` / `truth.parquet` to feature/label subsets).
  - Can be run independently via the command line (e.g., `python scripts/preprocessing.py --force`).
  - Is also imported by `data_loader.py` to seamlessly execute when data is missing.
  
## Reusable Modules (`src/`)
Files in `src/` contain core functionality to be aggregated by `main.py` or `scripts/`.
- **`src/data_loader.py`**: A centralized data interface. Other scripts read DataFrames through here (e.g., `load_total()`, `load_events()`). It automatically triggers `preprocessing.py` if datasets are missing.
- **`src/models.py`, `src/models/*.py`**: Architecture definitions.
- **`src/trainer.py`**: Model training loops and checkpointing. 
- **`src/plots.py`**: Chart generation and output logic.
- **`src/analyzer.py`**: Evaluation metrics calculation.

## Output Management (`outputs/`)
All dynamically generated assets (model weights, report graphs, evaluation logs) belong in `outputs/`.
- `main.py` and `scripts` must write their execution output here.
- This directory is generally excluded from version control.

# 1. Overall Architecture

At the top level, the repository is organized into the following key directories and files:

- **`data/`**: Contains raw and processed datasets.
- **`notebooks/`**: Jupyter notebooks for exploratory data analysis or experimentation.
- **`outputs/`**: Generated files such as model weights, logs, and reports.
- **`scripts/`**: Standalone scripts for specific tasks.
- **`src/`**: Source code for the project, including data loading, model definition, and training logic.
- **`main.py`**: The main entry point for running the machine learning pipeline.
- **`config.yaml`**: The source of truth for paths, hyperparameters, and application-level configuration.

## 2. In-File Code Structure
To maintain readability, every Python file should follow a standardized top-to-bottom layout:

1. **Module Docstring**: A brief explanation of the file's purpose.
2. **Imports**: Grouped into three distinct blocks (Standard library, 3rd party, Local).
3. **Logger Setup**: Grabbing the logger instance for the specific module.
4. **Constants**: Any file-specific static variables or configuration mappings.
5. **Core Logic**: Classes and Functions.
6. **Main block**: (`if __name__ == "__main__":`) - **Only** needed in `main.py` and files inside `scripts/`.

### Example File Template
```python
"""
data_loader.py: Handles loading and initial validation of raw datasets.
"""

# 1. Standard library imports
import os
import logging
from pathlib import Path

# 2. Third-party imports
import pandas as pd
import yaml

# 3. Local application imports
from src import helpers

# 4. Logger Setup
logger = logging.getLogger(__name__)

# 5. Constants
DEFAULT_ENCODING = "utf-8"

# 6. Core Logic
def load_data(filepath: str) -> pd.DataFrame:
    """Loads data from the specified path."""
    logger.debug(f"Attempting to load data from {filepath}")
    # ... logic ...
    logger.info("Data loaded successfully.")
    return pd.DataFrame()

# 7. Main Block (If applicable - typically only in main.py or scripts/)
if __name__ == "__main__":
    load_data("data/raw.csv")
```

## 3. Logger Helpers Usage
Proper logging is critical over using `print()` statements so everything can be tracked sequentially.

- **Setup / Initialization**: 
  - `main.py` and your `scripts/*.py` should invoke the root logger configurator at the very beginning (e.g., setting formatters, stdout, and file handlers) before calling any logic.
  - Every other file should just retrieve its own logger near the top: `logger = logging.getLogger(__name__)`.
- **Usage Guidelines**:
  - `logger.debug(...)`: Use for granular, step-by-step variable checks (e.g., "Loop iteration 45, value = X").
  - `logger.info(...)`: Use for general progress that the user *should* see (e.g., "Starting model training", "Finished data preprocessing").
  - `logger.warning(...)`: Use when something isn't right, but the application can safely proceed.
  - `logger.error(...)` / `logger.exception(...)`: Use when an operation fails. `exception` automatically appends the stack trace.