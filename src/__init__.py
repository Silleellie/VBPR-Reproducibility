import os
from pathlib import Path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = str(Path(os.path.join(THIS_DIR, "..")).resolve())

DATA_DIR = os.path.join(ROOT_PATH, "data")
RAW_DIR = os.path.join(ROOT_PATH, "data", "raw")
INTERIM_DIR = os.path.join(ROOT_PATH, "data", "interim")
PROCESSED_DIR = os.path.join(ROOT_PATH, "data", "processed")
MODEL_DIR = os.path.join(ROOT_PATH, "models")
REPORTS_DIR = os.path.join(ROOT_PATH, "reports")


# this will change if command arguments are passed to pipeline.py
class ExperimentConfig:
    epochs = [5, 10, 20, 50]
    batch_size = 128
    gamma_dim = 20
    theta_dim = 20
    lr = 0.005
    random_state = 42
    experiment = "comparison"
    num_threads = 4
