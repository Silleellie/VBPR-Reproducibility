"""
Module which contains util functions used by different modules throughout the experiment
"""

import csv
import os

from src import PROCESSED_DIR


def load_user_map():
    """
    Load the user map in the data/processed folder under the 'user_map.csv' name

    Returns:
        - user_map: dictionary containing the user_id as keys and the user_idx as values
    """
    user_map = {}
    with open(os.path.join(PROCESSED_DIR, "user_map.csv"), "r", encoding='utf-8') as file:

        iterat = csv.DictReader(file)

        for line in iterat:
            user_map[line["user_id"]] = int(line["user_idx"])

    return user_map


def load_item_map():
    """
    Load the item map in the data/processed folder under the 'item_map.csv' name

    Returns:
        - item_map: dictionary containing the item_id as keys and the item_idx as values
    """
    item_map = {}
    with open(os.path.join(PROCESSED_DIR, "item_map.csv"), "r", encoding='utf-8') as file:
        iterat = csv.DictReader(file)

        for line in iterat:
            item_map[line["item_id"]] = int(line["item_idx"])

    return item_map


def load_train_test_instances(mode: str = "train"):
    """
    Load instances for either the train or test set

    The instances will be triples following this format: (user_id, item_id, 1.0) [USER, ITEM, SCORE]

    Args:
        mode: either "train" or "test"

    Returns:
        tuples_instances: list containing the tuples
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
