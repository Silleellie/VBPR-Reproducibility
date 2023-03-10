from src.evaluation.ttest import main as stat_main


def main():
    print(" Computing AUC for ClayRS and Cornac ".center(80, '#'))
    print()
    from src.evaluation.compute_auc import main as auc_main  # perform import here just for pretty printing
    auc_main()
    print()
    print()

    print(" Computing ttest to compare AUC results ".center(80, '#'))
    print()
    stat_main()
    print()
    print()


if __name__ == "__main__":
    main()
