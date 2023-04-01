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
    # ---------- delete interactions for which we don't have feature (image doesn't exist) ----------
    tradesy_feedback = defaultdict(list)

    with open(path_item_features_csv_map, "r") as f:
        iterat = csv.DictReader(f)
        available_items_id = set(line["item_id"] for line in iterat)

    count_skipped_interactions = 0
    with open(path_positive_interactions, "r") as f:
        n_lines = sum(1 for _ in f)
        f.seek(0)

        iterat = csv.DictReader(f)
        for line in tqdm(iterat, total=n_lines - 1, desc="Splitting interactions in train/test"):
            user_id = line["uid"]
            pos_item_id = line["iid"]
            if pos_item_id in available_items_id:
                tradesy_feedback[user_id].append(pos_item_id)
            else:
                count_skipped_interactions += 1

    print(f"{count_skipped_interactions} interactions were skipped because involved missing items")

    return tradesy_feedback


def get_user_map(valid_positive_interactions: Dict[str, list]):

    user_map = {}
    for user_id in valid_positive_interactions:
        if user_id not in user_map:
            user_map[user_id] = len(user_map)

    return user_map


def get_train_test(tradesy_feedback: Dict[str, list]):

    train_feedback = {}
    test_feedback = {}

    # ---------- leave one out split ----------
    users_with_1_sample = 0
    for u in tradesy_feedback:
        user_pos_items = tradesy_feedback[u]
        try:
            train_items, test_items = train_test_split(user_pos_items, test_size=1,
                                                       random_state=ExperimentConfig.random_state)
            train_feedback[u] = train_items
            test_feedback[u] = test_items
        except ValueError:
            users_with_1_sample += 1

    print(f"{users_with_1_sample} users were skipped because they had less than one interaction "
          f"(thus they couldn't be split in train/test)")

    return train_feedback, test_feedback


def save_to_csv(train_dict, test_dict, user_map):
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
    print(f"Train set saved into {train_set_fname}")

    user_map_df.to_csv(user_map_fname, index=False)
    print(f"User map saved into {train_set_fname}")


def main():

    path_positive_interactions = os.path.join(INTERIM_DIR, "filtered_positive_interactions_tradesy.csv")
    path_item_id_features_map = os.path.join(PROCESSED_DIR, "item_map.csv")

    valid_positive_interactions = filter_interactions_wo_features(path_positive_interactions, path_item_id_features_map)

    train_feedback_dict, test_feedback_dict = get_train_test(valid_positive_interactions)

    # all users in train appear in test set, so we can sefely use this
    user_map = get_user_map(train_feedback_dict)

    save_to_csv(train_feedback_dict, test_feedback_dict, user_map)


if __name__ == "__main__":
    main()
