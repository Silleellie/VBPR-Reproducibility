import os
from typing import Dict, Set

from tqdm import tqdm
from PIL import Image
import io
import numpy as np
import pandas as pd

from src import INTERIM_DIR, PROCESSED_DIR, RAW_DIR


def extract_images_from_npy(ds_path: str, item_ids_to_extract: Set[str]):

    print("Loading NPY matrix containing all tradesy images...")

    images_paths_csv = os.path.join(INTERIM_DIR, 'tradesy_images_paths.csv')
    tradesy_images_dir = os.path.join(INTERIM_DIR, 'tradesy_images')

    if not os.path.isfile(ds_path):
        raise FileNotFoundError(f"Couldn't find images dataset in specified path {ds_path}")

    if os.path.isdir(tradesy_images_dir) and os.path.isfile(images_paths_csv):
        print(f"Tradesy images dataset was already extracted into {ds_path}, skipped")
        print(f"Tradesy images paths csv was already extracted into {images_paths_csv}, skipped")
    else:
        _, _, _, items_images, _, _ = np.load(ds_path, allow_pickle=True, encoding='bytes')

        if not os.path.isdir(tradesy_images_dir):
            os.makedirs(tradesy_images_dir)

        csv_raw_source = 'itemID,image_path\n'

        successfully_extracted_imgs = 0

        for img_info in tqdm(list(items_images.values()), desc="Extracting only relevant images for the experiment..."):

            item_id = img_info[b'asin'].decode()
            if item_id in item_ids_to_extract:

                img = img_info[b'imgs']

                img_path = os.path.join(tradesy_images_dir, f'{item_id}.jpg')
                img = Image.open(io.BytesIO(img))
                img.save(img_path)

                csv_raw_source += f'{item_id},{os.path.join("tradesy_images", f"{item_id}.jpg")}\n'

                successfully_extracted_imgs += 1

        with open(images_paths_csv, 'w') as f:
            f.write(csv_raw_source)

        print()
        print(f"{successfully_extracted_imgs}/{len(item_ids_to_extract)} images extracted and saved "
              f"into {tradesy_images_dir}!")
        print(f"CSV containing image id and relative path of each img saved "
              f"into {tradesy_images_dir}!")


def main():

    train_interactions = pd.read_csv(os.path.join(PROCESSED_DIR, "train_set.csv"), dtype=str)
    test_interactions = pd.read_csv(os.path.join(PROCESSED_DIR, "test_set.csv"), dtype=str)

    items_id_to_extract = set(train_interactions["item_id"].append(test_interactions["item_id"], ignore_index=True))

    path_raw_source_npy = os.path.join(RAW_DIR, "TradesyImgPartitioned.npy")

    extract_images_from_npy(path_raw_source_npy, items_id_to_extract)


if __name__ == "__main__":
    main()