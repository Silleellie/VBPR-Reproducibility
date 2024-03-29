"""
Module used by `exp1`, `exp2` and `exp3` experiments.

It is used to convert the tradesy raw interaction dataset into a CSV following UIR (user-item-rating) format and to
preprocess it following the original VBPR paper instructions
"""

import ast
import itertools
import os
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

from src import RAW_DIR, INTERIM_DIR


def import_tradesy_feedback(path: str) -> dict:
    """
    Create a dictionary where keys are string user ids and values are lists of item ids that are positive
    items for that user from the .json file containing tne interactions

        ex:

            {
                "0": ["1", "52"],
                "1": ["10"],
                ...
            }

    The format of the .json file is the one provided by the VBPR authors ('tradesy.json')

        ex:

            {'lists': {'bought': [],'selling': [], 'want': [], 'sold': ['3', '2']}, 'uid': '1'}

    Args:
        path: path where the .json file is stored

    Returns:
        ratings_dict: dictionary where keys are user ids and values are list of positive item ids for each user

    """

    ratings_dict = defaultdict(list)

    # ----------- LOAD USER POSITIVE ITEMS IN DICT -----------

    with open(path, 'r', encoding='utf-8') as file:
        n_lines = sum(1 for _ in file)

        file.seek(0)
        for line in tqdm(file, desc="Importing raw feedbacks...", total=n_lines):
            user_profile = ast.literal_eval(line)

            user_id = user_profile["uid"]
            user_positive_items = user_profile["lists"]["want"] + user_profile["lists"]["bought"]

            ratings_dict[user_id].extend(user_positive_items)

    return ratings_dict


def preprocess_tradesy_feedback(tradesy_feedback: dict) -> pd.DataFrame:
    """
    Remove duplicate items in users interactions and then remove users which have less than 5 implicit ratings

    The resulting dictionary is also saved as a .csv file in the `processed` dir

    Args:
        tradesy_feedback: dictionary where keys are user ids and values are lists of item ids that are positive
            items for that user
            ex:

                {
                    "0": ["1", "52"],
                    "1": ["10"],
                    ...
                }

    Returns:
        implicit ratings: processed implicit ratings dataframe

    """

    filtered_dict = {user_id: np.unique(positive_items) for user_id, positive_items in tradesy_feedback.items()
                     if len(np.unique(positive_items)) >= 5}

    # ----------- SAVE PREPROCESSED DICT AS CSV BY FIRST CREATING THE PANDAS DATAFRAME -----------

    user_col = [user_id for user_id in filtered_dict for _ in range(len(filtered_dict[user_id]))]
    item_col = list(itertools.chain.from_iterable(filtered_dict.values()))
    positive_col = [1.0 for _ in range(0, len(item_col))]

    implicit_ratings = pd.DataFrame(
        {
            'uid': user_col,
            'iid': item_col,
            'positive': positive_col
        }
    )

    implicit_ratings.to_csv(os.path.join(INTERIM_DIR, "filtered_positive_interactions_tradesy.csv"), index=False)

    return implicit_ratings


def main():
    """
    Actual main function of the module.

    Raw tradesy feedback are first converted from JSON to CSV with UIR format (invoking `import_tradesy_feedback()`) and
    then preprocessed (invoking `preprocess_tradesy_feedback()`)

    """

    raw_tradesy_ratings = import_tradesy_feedback(os.path.join(RAW_DIR, "tradesy.json"))

    raw_interactions = list(itertools.chain.from_iterable(raw_tradesy_ratings.values()))
    raw_items = set(raw_interactions)

    print("Cardinality of raw dataset:")
    print(f"Number of users: {len(raw_tradesy_ratings)}")
    print(f"Number of items: {len(raw_items)}")
    print(f"Number of positive interactions: {len(raw_interactions)}")

    print("".center(80, '-'))

    processed_tradesy_ratings = preprocess_tradesy_feedback(raw_tradesy_ratings)

    n_users = len(processed_tradesy_ratings["uid"].unique())
    n_items = len(processed_tradesy_ratings["iid"].unique())
    n_interactions = len(processed_tradesy_ratings)

    print("Cardinality of preprocessed dataset:")
    print(f"Number of users: {n_users}")
    print(f"Number of items: {n_items}")
    print(f"Number of positive interactions: {n_interactions}")

    print()
    print(f"Preprocessed dataset saved into {os.path.join(INTERIM_DIR, 'filtered_positive_interactions_tradesy.csv')}!")


if __name__ == "__main__":
    main()
