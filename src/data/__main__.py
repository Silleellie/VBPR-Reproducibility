"""
Main module of the data preparation phase
"""

import os.path

from src.data.create_interaction_csv import main as csv_main
from src.data.create_tradesy_images_dataset import main as images_main
from src.data.dl_raw_sources import main_additional as dl_main_additional
from src.data.dl_raw_sources import main_comparison as dl_main_comparison
from src.data.extract_features_from_source import main as features_main
from src.data.train_test_split import main as split_main

from src import ExperimentConfig, PROCESSED_DIR


def main():
    """
    Main which performs the data preparation phase by calling functions w.r.t. the operations to carry out for the
    specified experiment type (comparison or additional)

    """

    print(" Downloading raw sources ".center(80, '#'))
    print()
    if ExperimentConfig.experiment == "comparison":
        dl_main_comparison()
    else:
        dl_main_additional()
    print()
    print()

    print(" Filtering positive interactions ".center(80, '#'))
    print()
    csv_main()
    print()
    print()

    if ExperimentConfig.experiment == "comparison":
        print(" Extracting visual features ".center(80, '#'))
        print()
        features_main()
        print()
        print()
    else:

        if not os.path.isfile(os.path.join(PROCESSED_DIR, "item_map.csv")):
            print(" Downloading original visual features to build the item map ".center(80, '#'))
            print()
            dl_main_comparison()
            print()
            print()
            print(" Extracting original visual features to build the item map ".center(80, '#'))
            print()
            features_main()
            print()
            print()

        print(" Extracting images dataset ".center(80, '#'))
        print()
        images_main()
        print()
        print()

    print(" Building train/test set ".center(80, '#'))
    print()
    split_main()
    print()
    print()


if __name__ == "__main__":
    main()
