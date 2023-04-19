"""
Module used by `exp1`, `exp2` and `exp3` experiments.

Computes the AUC metric considering models fit and serialized by both experiment:
    * Cornac and ClayRS models at 5, 10, 20, 50 epochs (`comparsion` experiment)
    * ClayRS models fit on two different Content Analyzer generated representations,
        using the caffe reference model for both but with different pre-processing operations, at 10, 20 epochs
    * ClayRS models fit on two different Content Analyzer generated representations,
        using the vgg19 and resnet50 pre-trained neural networks with their required pre-processing operations,
        at 10, 20 epochs
"""

import os
import pickle
import time
from typing import List, Tuple

import cornac
import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy_indexed as npi

from cornac.data import Dataset, ImageModality
from cornac.eval_methods import ranking_eval
import cornac.models.vbpr.recom_vbpr

import clayrs_can_see.content_analyzer as ca
import clayrs_can_see.recsys as rs

from src import MODEL_DIR, PROCESSED_DIR, REPORTS_DIR, ExperimentConfig
from src.utils import load_user_map, load_item_map, load_train_test_instances


def auc_cornac(vbpr_cornac: cornac.models.vbpr.recom_vbpr.VBPR, train_dataset: List[Tuple], test_dataset: List[Tuple]):
    """
    Compute AUC metric on Cornac model invoking `ranking_eval` function from the framework itself

    Args:
        vbpr_cornac: fit vbpr algorithm using the Cornac framework
        train_dataset: list containing the interaction tuples (representing the train set)
        test_dataset: list containing the interaction tuples (representing the test set)

    Returns:
        sys_results: average AUC over all users
        user_results: dataframe containing for each user integer key its corresponding AUC value

    """

    sys_result, users_results = ranking_eval(vbpr_cornac, [cornac.metrics.AUC()], train_dataset, test_dataset)

    sys_result = sys_result[0]
    users_results = users_results[0]

    users_results = pd.DataFrame({
        'user_idx': list(users_results.keys()),
        'AUC': list(users_results.values())
    })

    return sys_result, users_results


# pylint: disable=too-many-locals
def auc_clayrs(vbpr_clayrs: rs.ContentBasedRS, train_set: ca.Ratings, test_set: ca.Ratings):
    """
    Compute AUC metric on ClayRS model using a custom defined implementation of the AUC metric via the numpy library.

    Args:
        vbpr_clayrs: fit vbpr algorithm using the ClayRS framework
        train_set: list containing the interaction tuples (representing the train set)
        test_set: list containing the interaction tuples (representing the test set)

    Returns:
        sys_results: average AUC over all users
        user_results: dataframe containing for each user integer key its corresponding AUC value

    """

    n_items = len(test_set.item_map)
    item_idxs = np.arange(0, n_items)
    query_vector = np.full(item_idxs.shape, True)

    pbar = tqdm(test_set.user_idx_column)
    pbar.set_description("Computing AUC...")

    per_user_result = {"user_idx": [], "AUC": []}
    for user_idx in pbar:

        positive_items_idxs = train_set.item_idx_column[train_set.get_user_interactions(user_idx, as_indices=True)]
        query_vector[positive_items_idxs] = False

        filter_list = item_idxs[query_vector]

        predictions = vbpr_clayrs.fit_alg.return_scores(user_idx, filter_list)

        test_idx = npi.indices(filter_list,
                               test_set.item_idx_column[test_set.get_user_interactions(user_idx, as_indices=True)])

        test_predictions = predictions[test_idx]
        negative_predictions = np.delete(predictions, test_idx)

        user_auc = (test_predictions > negative_predictions).sum() / (len(negative_predictions) * len(test_predictions))

        per_user_result["user_idx"].append(user_idx)
        per_user_result["AUC"].append(user_auc)

        # reset for next cycle
        query_vector[positive_items_idxs] = True

        if len(per_user_result["AUC"]) % 100 == 0 or len(per_user_result["AUC"]) == len(test_set):
            pbar.set_description(f"AUC after evaluating {len(per_user_result['AUC'])}/{len(test_set)} users ---> "
                                 f"{sum(per_user_result['AUC']) / len(per_user_result['AUC']):.3f}")

    sys_result = np.nanmean(per_user_result["AUC"])
    per_user_result = pd.DataFrame(per_user_result)

    return sys_result, per_user_result


# pylint: disable=too-many-locals
def evaluate_clayrs(models_exp_dir: str, repr_id: str, epoch: str):
    """
    Evaluate the ClayRS model fit on the specified number of epochs and on the specified representation key
    by first loading it into memory together with the train and test set and invoke the `auc_clayrs()` method to
    compute the AUC metric

    Args:
        epoch: integer used to retrieve the corresponding ClayRS model trained on that number of epochs
        repr_id: string used to retrieve the corresponding ClayRS model trained on that content representation

    Returns:
        sys_results: dataframe containing the average AUC value over all users and the amount of time required
            by the evaluation
        user_results: dataframe containing for each user integer key its corresponding AUC value

    """

    with open(os.path.join(models_exp_dir, f"vbpr_clayrs_{repr_id}_{epoch}.ml"), "rb") as file:
        rec_sys = pickle.load(file)

    user_map = load_user_map()
    item_map = load_item_map()

    train_tuples = load_train_test_instances(mode="train")
    test_tuples = load_train_test_instances(mode="test")

    train_set = ca.Ratings.from_list(train_tuples, user_map=user_map, item_map=item_map)
    test_set = ca.Ratings.from_list(test_tuples, user_map=user_map, item_map=item_map)

    start = time.time()
    sys_result, users_result = auc_clayrs(rec_sys, train_set, test_set)
    end = time.time()

    elapsed_m, elapsed_s = divmod(end - start, 60)

    sys_result = pd.DataFrame({
        "AUC": [sys_result],
        "Elapsed time": [f"{int(elapsed_m)}m {int(elapsed_s)}s"]
    })

    return sys_result, users_result


# pylint: disable=too-many-locals
def evaluate_cornac(models_exp_dir: str, epoch: int):
    """
    Evaluate the Cornac model fit on the specified number of epochs by first loading it into memory together with the
    train and test set and invoke the `auc_cornac()` method to compute the AUC metric

    Args:
        epoch: integer used to retrieve the corresponding Cornac model trained on that number of epochs

    Returns:
        sys_results: dataframe containing the average AUC value over all users and the amount of time required
            by the evaluation
        user_results: dataframe containing for each user integer key its corresponding AUC value

    """

    with open(os.path.join(models_exp_dir, f"vbpr_cornac_{epoch}.ml"), "rb") as file:
        vbpr_cornac = pickle.load(file)

    user_map = load_user_map()
    item_map = load_item_map()

    train_tuples = load_train_test_instances(mode="train")
    test_tuples = load_train_test_instances(mode="test")

    features_matrix = np.load(os.path.join(PROCESSED_DIR, "features_matrix.npy"))

    train_dataset = Dataset.build(train_tuples, global_uid_map=user_map, global_iid_map=item_map)
    test_dataset = Dataset.build(test_tuples, global_uid_map=user_map, global_iid_map=item_map)

    # mock iterator to disable shuffle for replicability
    train_dataset.uij_iter = lambda batch_size, shuffle: Dataset.uij_iter(train_dataset, batch_size, shuffle=False)
    test_dataset.uij_iter = lambda batch_size, shuffle: Dataset.uij_iter(test_dataset, batch_size, shuffle=False)

    # Instantiate a ImageModality for the two datasets
    item_image_modality = ImageModality(features=features_matrix, ids=list(item_map.keys()), normalized=True)
    item_image_modality.build()

    train_dataset.add_modalities(item_image=item_image_modality)
    test_dataset.add_modalities(item_image=item_image_modality)

    vbpr_cornac.train_set = train_dataset

    start = time.time()
    sys_result, users_result = auc_cornac(vbpr_cornac, train_dataset, test_dataset)
    end = time.time()

    elapsed_m, elapsed_s = divmod(end - start, 60)

    sys_result = pd.DataFrame({
        "AUC": [sys_result],
        "Elapsed time": [f"{int(elapsed_m)}m {int(elapsed_s)}s"]
    })

    return sys_result, users_result


def common_eval_clayrs(models_exp_dir: str, field_representation_list: list, results_output_dir: str):
    """
    Encapsulates the common operations carried out to compute the AUC metric on models trained using the
    ClayRS framework.
    The AUC results will be stored in dataframes and saved locally using the following formats:

        * "sys_result_clayrs_{repr_id}_{epoch_num}.csv"
        * "users_results_clayrs_{repr_id}_{epoch_num}.csv"

    Each result will be uniquely identified by the Content Analyzer representation that was used to train the model and
    the number of training epochs

    Args:
        models_exp_dir: path to the directory where the trained models are stored
        field_representation_list: list containing the id of each representation to take into account during evaluation
        results_output_dir: path to the directory where the results of the evaluation will be stored

    """

    print("".center(80, "*"))
    for repr_id in field_representation_list:

        print("Considering representation: ", repr_id)
        print("".center(80, "*"))

        for epoch_num in ExperimentConfig.epochs:

            print(f"Considering number of epochs {epoch_num}")
            print("".center(80, '-'))

            sys_result_clayrs, users_results_clayrs = evaluate_clayrs(models_exp_dir, repr_id, epoch_num)

            print(f"AUC: {float(sys_result_clayrs['AUC'][0])}, "
                  f"Elapsed time: {str(sys_result_clayrs['Elapsed time'][0])}\n")

            sys_result_output_path = os.path.join(results_output_dir,
                                                  f"sys_result_clayrs_{repr_id}_{epoch_num}.csv")
            users_results_output_path = os.path.join(results_output_dir,
                                                     f"users_results_clayrs_{repr_id}_{epoch_num}.csv")

            sys_result_clayrs.to_csv(sys_result_output_path,
                                     index=False)
            users_results_clayrs.to_csv(users_results_output_path,
                                        index=False)

            print(f"AUC sys results saved into {sys_result_output_path}!")
            print(f"AUC per user results saved into {users_results_output_path}!")

            # if this is the last epoch we do not print the epoch separator ("-")
            if epoch_num != ExperimentConfig.epochs[-1]:
                print("".center(80, '-'))

        # if this is the last representation to use we do not print the representation separator ("*")
        if repr_id != field_representation_list[-1]:
            print("".center(80, "*"))


def main_exp1():
    """
    Actual main function of the module for the `exp1` experiment.

    It will compute the AUC metric system-wise and for each user considering ClayRS and Cornac VBPR fit models on all
    number of epochs specified via the `-epo` cmd argument (invoking `evaluate_clayrs()`, `evaluate_cornac()`).

    Results will be saved into `reports/results_clayrs` and `reports/results_cornac`.

    """

    print("Evaluating ClayRS:")
    print("".center(80, "-"))

    models_clayrs_dir = os.path.join(MODEL_DIR, "exp1", "vbpr_clayrs")
    models_cornac_dir = os.path.join(MODEL_DIR, "exp1", "vbpr_cornac")

    results_clayrs_dir = os.path.join(REPORTS_DIR, "results_clayrs")
    results_cornac_dir = os.path.join(REPORTS_DIR, "results_cornac")

    os.makedirs(results_clayrs_dir, exist_ok=True)
    os.makedirs(results_cornac_dir, exist_ok=True)

    common_eval_clayrs(models_exp_dir=models_clayrs_dir,
                       field_representation_list=["imported_features"],
                       results_output_dir=results_clayrs_dir)

    print()
    print()
    print("Evaluating Cornac:")
    print("".center(80, "-"))
    for epoch in ExperimentConfig.epochs:
        print(f"Considering number of epochs {epoch}")
        print("".center(80, "-"))
        sys_result_cornac, users_results_cornac = evaluate_cornac(models_cornac_dir, epoch)

        print(f"AUC: {float(sys_result_cornac['AUC'][0])}, "
              f"Elapsed time: {str(sys_result_cornac['Elapsed time'][0])}\n")

        sys_result_cornac.to_csv(os.path.join(results_cornac_dir,
                                              f"sys_result_cornac_{epoch}.csv"), index=False)
        users_results_cornac.to_csv(os.path.join(results_cornac_dir,
                                                 f"users_results_cornac_{epoch}.csv"), index=False)

        print(f"AUC sys results saved into "
              f"{os.path.join(results_cornac_dir, f'sys_result_cornac_{epoch}.csv')}")
        print(f"AUC per user results saved into "
              f"{os.path.join(results_cornac_dir, f'users_results_cornac_{epoch}.csv')}")

        # if this is the last epoch we do not print the separator
        # pylint: disable=duplicate-code
        if epoch != ExperimentConfig.epochs[-1]:
            print("".center(80, '-'))


def main_exp2():
    """
    Actual main function of the module for the `exp2` experiment.

    It will compute the AUC metric system-wise and for each user considering ClayRS VBPR fit models on all
    number of epochs specified via the `-epo` cmd argument and on the representations identified by the following ids:
    'caffe' and 'caffe_center_crop' (invoking `evaluate_clayrs()`).

    Results will be saved into `reports/exp2`.

    """

    print("Evaluating ClayRS:")
    print("".center(80, "-"))

    models_clayrs_dir = os.path.join(MODEL_DIR, "exp2")

    results_clayrs_dir = os.path.join(REPORTS_DIR, "exp2")
    os.makedirs(results_clayrs_dir, exist_ok=True)

    field_representation_list = ["caffe", "caffe_center_crop"]

    common_eval_clayrs(models_exp_dir=models_clayrs_dir,
                       field_representation_list=field_representation_list,
                       results_output_dir=results_clayrs_dir)


def main_exp3():
    """
    Actual main function of the module for the `exp3` experiment.

    It will compute the AUC metric system-wise and for each user considering ClayRS VBPR fit models on all
    number of epochs specified via the `-epo` cmd argument and on the representations identified by the following ids:
    'vgg19' and 'resnet50' (invoking `evaluate_clayrs()`).

    Results will be saved into `reports/exp3`.

    """

    print("Evaluating ClayRS:")
    print("".center(80, "-"))

    models_clayrs_dir = os.path.join(MODEL_DIR, "exp3")

    results_clayrs_dir = os.path.join(REPORTS_DIR, "exp3")
    os.makedirs(results_clayrs_dir, exist_ok=True)

    field_representation_list = ["vgg19", "resnet50"]

    common_eval_clayrs(models_exp_dir=models_clayrs_dir,
                       field_representation_list=field_representation_list,
                       results_output_dir=results_clayrs_dir)


if __name__ == "__main__":

    # pylint: disable=duplicate-code
    if ExperimentConfig.experiment == "exp1":
        main_exp1()
    elif ExperimentConfig.experiment == "exp2":
        main_exp2()
    else:
        main_exp3()
