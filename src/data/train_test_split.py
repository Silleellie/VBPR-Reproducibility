"""
Module used both by `comparison` and `additional` experiment.

Possibly remove from the ratings items for which there are no visual features.
Build both the train/test split following the instruction of the VBPR experiment and the user mapping.
"""

import csv
import itertools
import os
from collections import defaultdict
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src import INTERIM_DIR, PROCESSED_DIR, ExperimentConfig


def filter_interactions_wo_features(path_positive_interactions: str, path_item_features_csv_map: str):
    """
    Remove from the ratings interactions involving items for which visual features are not available

    Args:
        path_positive_interactions: path where the tradesy filtered interactions .csv file is stored
        path_item_features_csv_map: path where the item map .csv file is stored

    Returns:
        tradesy_feedback: tradesy feedback without interactions involving items with no visual features

    """
    # ---------- delete interactions for which we don't have feature (image doesn't exist) ----------
    tradesy_feedback = defaultdict(list)

    with open(path_item_features_csv_map, "r", encoding='utf-8') as file:
        iterat = csv.DictReader(file)
        available_items_id = set(line["item_id"] for line in iterat)

    count_skipped_interactions = 0
    with open(path_positive_interactions, "r", encoding='utf-8') as file:
        n_lines = sum(1 for _ in file)
        file.seek(0)

        iterat = csv.DictReader(file)
        for line in tqdm(iterat, total=n_lines - 1, desc="Splitting interactions in train/test"):
            user_id = line["uid"]
            pos_item_id = line["iid"]
            if pos_item_id in available_items_id:
                tradesy_feedback[user_id].append(pos_item_id)
            else:
                count_skipped_interactions += 1

    print(f"{count_skipped_interactions} interactions were skipped because involved missing items")

    return tradesy_feedback


def build_user_map(valid_positive_interactions: Dict[str, list]):
    """
    Build mapping between user string ids and integer ids, the ordering is the same as the one in the positive
    interactions dictionary

    Args:
        valid_positive_interactions: dictionary containing the valid positive interactions for each user

            ex:
                {
                "0": ["1", "52"],
                "1": ["10"],
                ...
                }

    Returns:
        user_map: dictionary containing the mapping

            ex:
                {
                    "0": 0,
                    "1": 1,
                    ...
                }

    """

    user_map = {}
    for user_id in valid_positive_interactions:
        if user_id not in user_map:
            user_map[user_id] = len(user_map)

    return user_map


def build_train_test(tradesy_feedback: Dict[str, list]):
    """
    Build train and test set performing leave-one-out with fixed random state (set via -seed command line argument)

    Args:
        tradesy_feedback: dictionary with string user ids as keys and list of items involved in positive interactions
            as values for each user

    Returns:
        train_feedback: dictionary having user ids as keys and lists of item ids as values (representing the train set)
        test_feedback: dictionary having user ids as keys and lists of item ids as values (representing the test set)

    """

    train_feedback = {}
    test_feedback = {}

    # ---------- leave one out split ----------
    users_with_1_sample = 0
    for user in tradesy_feedback:
        user_pos_items = tradesy_feedback[user]
        try:
            train_items, test_items = train_test_split(user_pos_items, test_size=1,
                                                       random_state=ExperimentConfig.random_state)
            train_feedback[user] = train_items
            test_feedback[user] = test_items
        except ValueError:
            users_with_1_sample += 1

    print(f"{users_with_1_sample} users were skipped because they had less than one interaction "
          f"(thus they couldn't be split in train/test)")

    return train_feedback, test_feedback


def save_to_csv(train_dict, test_dict, user_map):
    """
    Save train, test and user map dictionaries as .csv files (loading them into a pandas DataFrame).
    All the resulting files will be saved into the `data/processed` directory.

    Args:
        train_dict: dictionary having user ids as keys and lists of item ids as values (representing the train set)
        test_dict: dictionary having user ids as keys and lists of item ids as values (representing the test set)
        user_map: dictionary having user ids as keys and user integer ids as values

    """

    train_df = pd.DataFrame({
        "user_id": [user_id for user_id in train_dict for _ in range(len(train_dict[user_id]))],
        "item_id": list(itertools.chain.from_iterable(train_dict.values())),
        "score": 1
    })

    test_df = pd.DataFrame({
        "user_id": [user_id for user_id in test_dict for _ in range(len(test_dict[user_id]))],
        "item_id": list(itertools.chain.from_iterable(test_dict.values())),
        "score": 1
    })

    user_map_df = pd.DataFrame({
        "user_id": list(user_map.keys()),
        "user_idx": list(user_map.values())
    })

    print()

    train_set_fname = os.path.join(PROCESSED_DIR, "train_set.csv")
    test_set_fname = os.path.join(PROCESSED_DIR, "test_set.csv")
    user_map_fname = os.path.join(PROCESSED_DIR, "user_map.csv")

    train_df.to_csv(train_set_fname, index=False)
    print(f"Train set saved into {train_set_fname}")

    test_df.to_csv(test_set_fname, index=False)
    print(f"Test set saved into {test_set_fname}")

    user_map_df.to_csv(user_map_fname, index=False)
    print(f"User map saved into {user_map_fname}")


def main():
    """
    Actual main function of the module.

    Preprocessed tradesy feedback are first filtered in case interactions involving items with no visual
    feature available are present (invoking `filter_interactions_wo_features()`), then train set, test set and user map
    are built and saved as .csv files (invoking `build_train_test()`, `build_user_map()`, `save_to_csv()`)

    """

    path_positive_interactions = os.path.join(INTERIM_DIR, "filtered_positive_interactions_tradesy.csv")
    path_item_id_features_map = os.path.join(PROCESSED_DIR, "item_map.csv")

    valid_positive_interactions = filter_interactions_wo_features(path_positive_interactions, path_item_id_features_map)

    train_feedback_dict, test_feedback_dict = build_train_test(valid_positive_interactions)

    # all users in train appear in test set, so we can sefely use this
    user_map = build_user_map(train_feedback_dict)

    save_to_csv(train_feedback_dict, test_feedback_dict, user_map)


if __name__ == "__main__":
    main()
