"""
Module used by `exp1`, `exp2` and `exp3` experiments.

Computes the ttest statistical test, using AUC user wise results, for all models evaluated.
"""

import os.path

import pandas as pd
from clayrs import evaluation as eva

from src import REPORTS_DIR, ExperimentConfig


def perform_ttest(sys1_user_results: pd.DataFrame,
                  sys2_user_results: pd.DataFrame,
                  epoch: int,
                  results_output_dir: str):
    """
    Encapsulates the common operations carried out to perform the ttest statistical test on models trained using the
    ClayRS framework.
    The results of the evaluation procedure will be stored in dataframes and saved locally using the following format:

        * "ttest_{epoch}.csv"

    Each result will be uniquely identified by the number of training epochs

    Args:
        sys1_user_results: DataFrame containing the user-wise results for the first system
        sys2_user_results: DataFrame containing the user-wise results for the second system
        epoch: integer which represents the number of training epochs that were used to train the models that produced
            the results
        results_output_dir: path to the directory where the results of the evaluation will be stored

    """

    result = eva.Ttest("user_idx").perform([sys1_user_results, sys2_user_results])

    print(result, "\n")

    result.to_csv(os.path.join(results_output_dir, f"ttest_{epoch}.csv"), index=False)
    print(f"ttest results saved into {os.path.join(results_output_dir, f'ttest_{epoch}.csv')}!")


def main_exp1():
    """
    Actual main function of the module for the `exp1` experiment.

    It will compute the ttest statistical test using ClayRS comparing, for each number of epochs used in the experiment,
    between ClayRS and Cornac user results.

    Results will be saved into `reports/ttest_results/exp1`.

    """

    ttest_dir = os.path.join(REPORTS_DIR, 'ttest_results')
    os.makedirs(ttest_dir, exist_ok=True)

    exp1_ttest_dir = os.path.join(ttest_dir, 'exp1')
    os.makedirs(exp1_ttest_dir, exist_ok=True)

    print("".center(80, "-"))
    for epoch in ExperimentConfig.epochs:
        print(f"Considering number of epochs {epoch}")
        print("".center(80, "-"))
        result_clayrs = pd.read_csv(os.path.join(REPORTS_DIR, "exp1", "results_clayrs",
                                                 f"users_results_clayrs_imported_features_{epoch}.csv"))
        result_cornac = pd.read_csv(os.path.join(REPORTS_DIR, "exp1", "results_cornac",
                                                 f"users_results_cornac_{epoch}.csv"))

        perform_ttest(sys1_user_results=result_clayrs,
                      sys2_user_results=result_cornac,
                      epoch=epoch,
                      results_output_dir=exp1_ttest_dir)

        # if this is the last epoch we do not print the separator
        if epoch != ExperimentConfig.epochs[-1]:
            print("".center(80, '-'))


def main_exp2():
    """
    Actual main function of the module for the `exp2` experiment.

    It will compute the ttest statistical test using ClayRS comparing, for each number of epochs used in the experiment,
    between ClayRS models trained on representations identified by ids `caffe` and `caffe_center_crop`.

    Results will be saved into `reports/ttest_results/exp2`.

    """

    ttest_dir = os.path.join(REPORTS_DIR, 'ttest_results')
    os.makedirs(ttest_dir, exist_ok=True)

    exp2_ttest_dir = os.path.join(ttest_dir, 'exp2')
    os.makedirs(exp2_ttest_dir, exist_ok=True)

    print("".center(80, "-"))
    for epoch in ExperimentConfig.epochs:
        print(f"Considering number of epochs {epoch}")
        print("".center(80, "-"))
        result_caffe = pd.read_csv(os.path.join(REPORTS_DIR, "exp2",
                                                f"users_results_clayrs_caffe_{epoch}.csv"))
        result_caffe_crop = pd.read_csv(os.path.join(REPORTS_DIR, "exp2",
                                                     f"users_results_clayrs_caffe_center_crop_{epoch}.csv"))

        perform_ttest(sys1_user_results=result_caffe,
                      sys2_user_results=result_caffe_crop,
                      epoch=epoch,
                      results_output_dir=exp2_ttest_dir)

        # if this is the last epoch we do not print the separator
        if epoch != ExperimentConfig.epochs[-1]:
            print("".center(80, '-'))


def main_exp3():
    """
    Actual main function of the module for the `exp3` experiment.

    It will compute the ttest statistical test using ClayRS comparing, for each number of epochs used in the experiment,
    between ClayRS models trained on representations identified by ids `vgg19` and `resnet50`.

    Results will be saved into `reports/ttest_results/exp3`.

    """

    ttest_dir = os.path.join(REPORTS_DIR, 'ttest_results')
    os.makedirs(ttest_dir, exist_ok=True)

    exp3_ttest_dir = os.path.join(ttest_dir, 'exp3')
    os.makedirs(exp3_ttest_dir, exist_ok=True)

    print("".center(80, "-"))
    for epoch in ExperimentConfig.epochs:
        print(f"Considering number of epochs {epoch}")
        print("".center(80, "-"))
        result_caffe = pd.read_csv(os.path.join(REPORTS_DIR, "exp3",
                                                f"users_results_clayrs_vgg19_{epoch}.csv"))
        result_caffe_crop = pd.read_csv(os.path.join(REPORTS_DIR, "exp3",
                                                     f"users_results_clayrs_resnet50_{epoch}.csv"))

        perform_ttest(sys1_user_results=result_caffe,
                      sys2_user_results=result_caffe_crop,
                      epoch=epoch,
                      results_output_dir=exp3_ttest_dir)

        # if this is the last epoch we do not print the separator
        if epoch != ExperimentConfig.epochs[-1]:
            print("".center(80, '-'))


if __name__ == "__main__":

    # pylint: disable=duplicate-code
    if ExperimentConfig.experiment == "exp1":
        main_exp1()
    elif ExperimentConfig.experiment == "exp2":
        main_exp2()
    else:
        main_exp3()
