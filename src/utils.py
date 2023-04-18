"""
Module which contains util functions used by different modules throughout the experiment
"""

import csv
import os
import random

import numpy as np
import torch

from src import PROCESSED_DIR, ExperimentConfig


def load_user_map():
    """
    Load the user map in the `data/processed` folder under the `user_map.csv` name as a dictionary,
    which contains the mapping between string ids of users (`user_id`) and their corresponding integers (`user_idx`)

    Returns:
        user_map: dictionary containing the `user_id` as keys and the `user_idx` as values
    """
    user_map = {}
    with open(os.path.join(PROCESSED_DIR, "user_map.csv"), "r", encoding='utf-8') as file:

        iterat = csv.DictReader(file)

        for line in iterat:
            user_map[line["user_id"]] = int(line["user_idx"])

    return user_map


def load_item_map():
    """
    Load the item map in the `data/processed` folder under the `item_map.csv` name as a dictionary,
    which contains the mapping between string ids of users (`item_id`) and their corresponding integers (`item_idx`)

    Returns:
        item_map: dictionary containing the `item_id` as keys and the `item_idx` as values
    """
    item_map = {}
    with open(os.path.join(PROCESSED_DIR, "item_map.csv"), "r", encoding='utf-8') as file:
        iterat = csv.DictReader(file)

        for line in iterat:
            item_map[line["item_id"]] = int(line["item_idx"])

    return item_map


def load_train_test_instances(mode: str = "train"):
    """
    Load interaction instances for either the train or test set, depending on the `mode` parameter.

    The instances will be triples following this format: (user_id, item_id, 1.0) [USER, ITEM, SCORE]

    Args:
        mode: either "train" or "test"

    Returns:
        tuples_instances: list containing the interaction tuples
    """

    if mode == "train":
        fname = os.path.join(PROCESSED_DIR, "train_set.csv")

    else:
        fname = os.path.join(PROCESSED_DIR, "test_set.csv")

    tuples_instances = []
    with open(fname, "r", encoding='utf-8') as file:

        iterat = csv.DictReader(file)

        for line in iterat:
            tuples_instances.append((str(line["user_id"]), str(line["item_id"]), 1.0))

    return tuples_instances


def seed_everything():
    """
    Function which fixes the random state of each library used by this repository with the seed
    specified when invoking `pipeline.py`

    Returns:
        The integer random state set via command line argument

    """

    # seed everything
    seed = ExperimentConfig.random_state
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    print(f"Random seed set as {seed}")

    return seed
