from src.data.create_interaction_csv import main as csv_main
from src.data.dl_raw_sources import main as dl_main
from src.data.extract_features_from_source import main as features_main
from src.data.train_test_split import main as split_main


def main():
    print(" Downloading raw sources ".center(80, '#'))
    print()
    dl_main()
    print()
    print()

    print(" Filtering positive interactions ".center(80, '#'))
    print()
    csv_main()
    print()
    print()

    print(" Extracting visual features ".center(80, '#'))
    print()
    features_main()
    print()
    print()

    print(" Building train/test set ".center(80, '#'))
    print()
    split_main()
    print()
    print()


if __name__ == "__main__":
    main()
