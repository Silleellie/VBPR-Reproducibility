"""
Module used by `exp1`, `exp2` and `exp3` experiments.

Build both the npy matrix containing original visual features and the item mapping.
"""

import os.path
import csv
import struct
import gc

from tqdm import tqdm
import numpy as np
import pandas as pd

from src import INTERIM_DIR, RAW_DIR, PROCESSED_DIR


def prepare_raw_source(path_b_tradesy: str, path_processed_csv: str, chunk: int = 10000):
    """
    Used to extract from the binary file (provided by the VBPR authors) the visual features of the items which
    appear in the filtered interaction tradesy feedback.
    The visual features are stored into a NPY matrix and a mapping between string item ids and integer ones is created.

    Args:
        path_b_tradesy: path where the binary file containing visual features is stored
        path_processed_csv: path where the .csv file containing the filtered tradesy interactions is stored
        chunk: number of items to vstack at once into the npy matrix

    """

    def read_image_features(path, items_set):
        with open(path, 'rb') as file:
            file.seek(0)
            while True:
                item_id = file.read(10)
                item_id = item_id.strip().decode()

                if item_id == '':
                    break

                if item_id not in items_set:
                    file.seek(4*4096, 1)  # skip items features from current position (4 position for a single value)
                    continue

                item_vector = []
                for _ in range(4096):
                    item_vector.append(struct.unpack('f', file.read(4))[0])  # struct unpack returns a tuple

                yield item_id, item_vector

    def useful_items(path_csv):
        with open(path_csv, "r", encoding='utf-8') as file:

            iterat = csv.DictReader(file)
            items_set = set(line["iid"] for line in iterat)

        return items_set

    items_to_get: set = useful_items(path_processed_csv)

    csv_dict = {"item_id": [], "item_idx": []}

    feature_matrix = None
    chunk_features = []
    for i, (item_id, item_feature) in enumerate(tqdm(read_image_features(path_b_tradesy, items_to_get),
                                                     desc="Extracting only useful features from binary source...",
                                                     total=len(items_to_get))):
        csv_dict['item_id'].append(item_id)
        csv_dict['item_idx'].append(i)

        chunk_features.append(item_feature)
        if (i + 1) % chunk == 0:
            if feature_matrix is None:
                feature_matrix = np.vstack(chunk_features)
            else:
                feature_matrix = np.vstack((feature_matrix, chunk_features))

            del chunk_features
            gc.collect()
            chunk_features = []

    if len(chunk_features) != 0:
        feature_matrix = np.vstack((feature_matrix, chunk_features))

        del chunk_features
        gc.collect()

    csv_filename = os.path.join(PROCESSED_DIR, "item_map.csv")
    features_filename = os.path.join(PROCESSED_DIR, "features_matrix.npy")

    print("Feature extracted from binary source!")
    print(f"Total items extracted: {feature_matrix.shape[0]}/{len(items_to_get)}")
    print()

    pd.DataFrame(csv_dict).to_csv(csv_filename, index=False)
    print(f"CSV containing mapping between item ids and features into {csv_filename}!")

    np.save(features_filename, feature_matrix)
    print(f"NPY feature matrix saved into {features_filename}!")


def main():
    """
    Actual main function of the module.

    It will create the npy matrix and the item mapping (invoking `prepare_raw_source()`)

    """

    path_csv_interim = os.path.join(INTERIM_DIR, "filtered_positive_interactions_tradesy.csv")
    path_raw_source_b = os.path.join(RAW_DIR, "image_features_tradesy.b")

    prepare_raw_source(path_raw_source_b, path_csv_interim)


if __name__ == "__main__":
    main()
