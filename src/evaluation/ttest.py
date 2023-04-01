import os.path

import pandas as pd
from clayrs_can_see import evaluation as eva

from src import REPORTS_DIR, ExperimentConfig


def main_comparison():

    ttest_dir = os.path.join(REPORTS_DIR, 'ttest_results')
    os.makedirs(ttest_dir, exist_ok=True)

    print("".center(80, "-"))
    for epoch in ExperimentConfig.epochs:
        print(f"Considering number of epochs {epoch}")
        print("".center(80, "-"))
        results_clayrs = pd.read_csv(os.path.join(REPORTS_DIR, "results_clayrs", f"users_results_clayrs_{epoch}.csv"))
        results_cornac = pd.read_csv(os.path.join(REPORTS_DIR, "results_cornac", f"users_results_cornac_{epoch}.csv"))

        result = eva.Ttest("user_idx").perform([results_clayrs, results_cornac])

        print(result, "\n")

        result.to_csv(os.path.join(ttest_dir, f"ttest_{epoch}.csv"), index=False)
        print(f"ttest results saved into {os.path.join(ttest_dir, f'ttest_{epoch}.csv')}!")

        # if this is the last epoch we do not print the separator
        if epoch != ExperimentConfig.epochs[-1]:
            print("".center(80, '-'))


def main_additional():

    ttest_dir = os.path.join(REPORTS_DIR, 'ttest_results')
    os.makedirs(ttest_dir, exist_ok=True)

    repr_ids = ['resnet50', 'caffe', 'caffe_center_crop']
    results_additional_exp_dir = os.path.join(REPORTS_DIR, "results_additional_exp")

    for epoch in ExperimentConfig.epochs:
        results = [
            pd.read_csv(os.path.join(results_additional_exp_dir, f"users_results_additional_exp_{repr_id}_{epoch}.csv"))
            for repr_id in repr_ids]

        result = eva.Ttest("user_idx").perform(results)

        print(result, "\n")

        result.to_csv(os.path.join(ttest_dir, f"ttest_additional_exp_{epoch}.csv"), index=False)
        print(f"ttest results saved into {os.path.join(ttest_dir, f'ttest_{epoch}.csv')}!")

        # if this is the last epoch we do not print the separator
        if epoch != ExperimentConfig.epochs[-1]:
            print("".center(80, '-'))


if __name__ == "__main__":

    if ExperimentConfig.experiment == "comparison":
        main_comparison()
    else:
        main_additional()
