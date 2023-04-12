from src import ExperimentConfig
from src.evaluation.ttest import main_additional as stat_main_additional
from src.evaluation.ttest import main_comparison as stat_main_comparison


def main():

    if ExperimentConfig.experiment == "comparison":
        print(" Computing AUC for ClayRS and Cornac ".center(80, '#'))
        print()
        # perform import here just for pretty printing
        from src.evaluation.compute_auc import main_comparison as auc_main_comparison
        auc_main_comparison()
        print()
        print()

        print(" Computing ttest to compare AUC results ".center(80, '#'))
        print()
        stat_main_comparison()
        print()
        print()

    else:
        print(" Computing AUC for all different representations ".center(80, '#'))
        print()
        # perform import here just for pretty printing
        from src.evaluation.compute_auc import main_additional as auc_main_additional
        auc_main_additional()
        print()
        print()

        print(" Computing ttest to compare AUC results ".center(80, '#'))
        print()
        stat_main_additional()
        print()
        print()


if __name__ == "__main__":
    main()
