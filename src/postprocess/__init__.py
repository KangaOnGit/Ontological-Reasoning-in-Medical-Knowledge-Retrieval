from pathlib import Path
import yaml

# Automatically Recognizes Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    with open(PROJECT_ROOT / path, "r") as f:
        return yaml.safe_load(f)