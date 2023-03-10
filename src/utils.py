import csv
import os

from src import PROCESSED_DIR


def load_user_map():
    user_map = {}
    with open(os.path.join(PROCESSED_DIR, "user_map.csv"), "r") as f:

        iterat = csv.DictReader(f)

        for line in iterat:
            user_map[line["user_id"]] = int(line["user_idx"])

    return user_map


def load_item_map():
    item_map = {}
    with open(os.path.join(PROCESSED_DIR, "item_map.csv"), "r") as f:
        iterat = csv.DictReader(f)

        for line in iterat:
            item_map[line["item_id"]] = int(line["item_idx"])

    return item_map


def load_train_test_instances(mode: str = "train"):

    if mode == "train":
        fname = os.path.join(PROCESSED_DIR, "train_set.csv")

    else:
        fname = os.path.join(PROCESSED_DIR, "test_set.csv")

    tuples_instances = []
    with open(fname, "r") as f:

        iterat = csv.DictReader(f)

        for line in iterat:
            tuples_instances.append((str(line["user_id"]), str(line["item_id"]), 1.0))

    return tuples_instances
