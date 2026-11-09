from pathlib import Path
import yaml

# Define required keys hierarchically to match your nested YAML structure
REQUIRED_KEYS = {
    "raw_input": {"dir", "events", "truth"},
    "processed_output": {
        "dir",
        "total_features",
        "total_labels",
        "events_features",
        "events_labels",
        "false_events_features",
        "false_events_labels",
        "parameters",
    },
    "pairing": {"PAIR_THRESHOLD_NS", "n_off"},
    "detector": {"RADIUS_M", "HALF_HEIGHT_M"},
    "performance": {"FLUSH_EVERY"},
}

def get_repo_root() -> Path:
    """Finds the repository root dynamically using Path(__file__)."""
    # Assuming this file is at src/reader.py (2 levels down from root)
    return Path(__file__).resolve().parents[1]


