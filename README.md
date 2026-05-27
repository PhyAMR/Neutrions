# Machine Learning Project

This project uses `uv` for environment management.

## Directory Structure and Organization Guidelines

- **`config.yaml`**: The master configuration file. All hyperparameters, file paths, and general settings should live here. Avoid hardcoding values in scripts.
- **`main.py`**: The single entry point to run the pipeline. It orchestrates the flow from data loading to training and evaluation.

### `data/`
Contains all datasets. **Note: Large data files should be added to `.gitignore`.**
- `raw/`: The immutable, original data (like your initial `training` set). Never modify scripts to overwrite these files.
- `processed/`: Cleaned and transformed data ready for modeling. Should always be reproducible from the raw data via your scripts.

### `src/`
Contains all source code for the project. Keep logic modular here.
- `__init__.py`: Makes the src directory a Python package.
- `data_loader.py`: Handles data ingestion, cleaning, and preprocessing.
- `preprocessing.py`: Additional preprocessing steps.
- `models.py`: Defines neural network architectures and other model classes.
- `trainer.py`: Contains training/validation loops and handles model checkpointing.
- `analyzer.py`: Generates evaluation metrics, plots, and reports.
- `plots.py`: Contains functions for creating various types of plots.
- `cylinder.py`: Helper script for cylinder-based geometries or related logic.
- `models/`: Contains individual scripts/modules per model used:
  - `cnn_model.py`: Convolutional Neural Network formulation.
  - `graphnn.py`: Graph Neural Network formulation.

### `test/`
Contains test scripts and exploratory code runs.
- `load_raw.py`: Script to explore loading raw data.

### `outputs/`
Stores all generated artifacts from script executions. **These folders should generally be ignored by version control.**
- `weights/`: Trained model files (e.g., `.pt`, `.pth`, `.safetensors`).
- `plots/`: Visualizations like loss curves, sample predictions, and confusion matrices.
- `reports/`: Evaluation summaries, performance tables, and CSVs.
