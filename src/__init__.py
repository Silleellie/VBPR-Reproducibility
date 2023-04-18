"""
Init containing all required paths in the experiment (paths to the folders where data is stored for example)

It also contains the ExperimentConfig class where all parameters that can be set via command line are stored
"""

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
YAML_DIR = os.path.join(ROOT_PATH, "reports", "yaml_clayrs")


# pylint: disable=too-few-public-methods
class ExperimentConfig:
    """
    This class contains all modifiable attributes related to the experiment

        - epochs: list of number of epochs, a new recommender instance will be created for each defined number
        - batch_size: batch size that will be used during recommender training
        - gamma_dim: VBPR parameter
        - theta_dim: VBPR parameter
        - lr: learning rate that will be used during recommender training
        - random_state: seed that will be set for any library whenever random operations occur
        - experiment: either "comparison" or "additional"
        - num_threads: number of threads that will be used to serialize produced contents by the Content Analyzer module
            in ClayRS
    """
    epochs = [5, 10, 20, 50]
    batch_size = 128
    gamma_dim = 20
    theta_dim = 20
    lr = 0.005
    random_state = 42
    experiment = "comparison"
    num_threads = 4
