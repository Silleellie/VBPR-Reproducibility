"""
Main module of the evaluation phase
"""

from src import ExperimentConfig
from src.evaluation.ttest import main_additional as stat_main_additional
from src.evaluation.ttest import main_comparison as stat_main_comparison


def main():
    """
    Main which performs the evaluation phase by calling functions w.r.t. the operations to carry out for the
    specified experiment type (comparison or additional)

    """

    if ExperimentConfig.experiment == "comparison":
        print(" Computing AUC for ClayRS and Cornac ".center(80, '#'))
        print()
        # perform import here just for pretty printing
        # pylint: disable=import-outside-toplevel
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
        # pylint: disable=import-outside-toplevel
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
