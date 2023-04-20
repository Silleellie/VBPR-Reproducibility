"""
Main module of the evaluation phase
"""

from src import ExperimentConfig
from src.evaluation.ttest import main_exp1 as stat_main_exp1
from src.evaluation.ttest import main_exp2 as stat_main_exp2
from src.evaluation.ttest import main_exp3 as stat_main_exp3


def main():
    """
    Main which performs the data preparation phase by calling functions w.r.t. the operations to carry out for the
    specified experiment (exp1, exp2 or exp3)

    """

    if ExperimentConfig.experiment == "exp1":
        print(" Computing AUC for ClayRS and Cornac ".center(80, '#'))
        print()
        # perform import here just for pretty printing
        # pylint: disable=import-outside-toplevel
        from src.evaluation.compute_auc import main_exp1 as auc_main_exp1
        auc_main_exp1()
        print()
        print()

        print(" Computing ttest to compare AUC results ".center(80, '#'))
        print()
        stat_main_exp1()
        print()
        print()

    elif ExperimentConfig.experiment == "exp2":
        print(" Computing AUC for caffe and caffe_center_crop representations ".center(80, '#'))
        print()
        # perform import here just for pretty printing
        # pylint: disable=import-outside-toplevel
        from src.evaluation.compute_auc import main_exp2 as auc_main_exp2
        auc_main_exp2()
        print()
        print()

        print(" Computing ttest to compare AUC results ".center(80, '#'))
        print()
        stat_main_exp2()
        print()
        print()

    else:
        print(" Computing AUC for vgg19 and resnet50 representations ".center(80, '#'))
        print()
        # perform import here just for pretty printing
        # pylint: disable=import-outside-toplevel
        from src.evaluation.compute_auc import main_exp3 as auc_main_exp3
        auc_main_exp3()
        print()
        print()

        print(" Computing ttest to compare AUC results ".center(80, '#'))
        print()
        stat_main_exp3()
        print()
        print()


if __name__ == "__main__":
    main()
