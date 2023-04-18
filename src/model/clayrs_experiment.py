"""
Module used by the `comparison` experiment.

Uses the NPY feature matrix built in the data preparation phase to fit the VBPR algorithm using the ClayRS framework.
"""

import os
import pickle

import clayrs_can_see.content_analyzer as ca
import clayrs_can_see.recsys as rs
from clayrs_can_see.utils import Report

from src import PROCESSED_DIR, DATA_DIR, MODEL_DIR, ExperimentConfig, YAML_DIR
from src.utils import load_user_map, load_item_map


def content_analyzer_tradesy(tradesy_item_map_path: str, features_matrix_path: str, output_contents_dir: str):
    """
    Performs the Content Analyzer phase of the `comparison` experiment using the ClayRS framework.
    The representation that will be generated simply imports the original feature vectors and associates them
    to the corresponding item (thanks to the `FromNPY()` content technique).

    A .yml file containing the specified technique and its parameters is saved into the `reports/yaml_clayrs`
    directory.

    Args:
        tradesy_item_map_path: path where the item mapping is stored
        features_matrix_path: path where the .npy feature matrix is stored
        output_contents_dir: path to the directory where the contents will be serialized

    """

    items_ca = ca.ItemAnalyzerConfig(
        source=ca.CSVFile(tradesy_item_map_path),
        id="item_id",
        output_directory=output_contents_dir
    )

    items_ca.add_single_config(
        "item_idx",
        ca.FieldConfig(
            ca.FromNPY(
                npy_file_path=features_matrix_path
            )
        )
    )

    content_a = ca.ContentAnalyzer(items_ca, n_thread=ExperimentConfig.num_threads_ca)
    content_a.fit()

    Report(output_dir=YAML_DIR, ca_report_filename="ca_report_comparison_exp").yaml(content_analyzer=content_a)

    print()
    print(f"Output of the Content Analyzer saved into {output_contents_dir}!")
    print(f"Report of the Content Analyzer phase saved into {os.path.join(YAML_DIR, 'ca_report_comparison_exp.yml')}!")


def recsys_tradesy(train_set_path: str, items_dir: str, epoch: int):
    """
    Performs the Recommender System phase of the `comparison` experiment using the ClayRS framework.

    A recommender is fit for the specified number of epochs and then saved in the `models/vbpr_clayrs` directory.

    A .yml file containing the VBPR algorithm definition with its parameters is saved into the
    `reports/yaml_clayrs/rs_report_comparison_exp` directory.

    Args:
        train_set_path: path to where the train .csv file is stored
        items_dir: path to the directory where the serialized contents are stored
        epoch: number of epochs to train the recommender for

    """

    user_map = load_user_map()
    item_map = load_item_map()

    train_set = ca.Ratings(ca.CSVFile(train_set_path), user_map=user_map, item_map=item_map)

    item_field = {'item_idx': 0}

    # pylint: disable=duplicate-code
    alg = rs.VBPR(item_field, device='cuda:0',
                  epochs=epoch,
                  gamma_dim=ExperimentConfig.gamma_dim,
                  theta_dim=ExperimentConfig.theta_dim,
                  batch_size=ExperimentConfig.batch_size,
                  learning_rate=ExperimentConfig.lr,
                  seed=ExperimentConfig.random_state,
                  threshold=0,
                  normalize=True)

    cbrs = rs.ContentBasedRS(alg, train_set, items_dir)

    cbrs.fit()

    return cbrs


def main():
    """
    Actual main function of the module.

    It first serializes the contents complexly represented (invoking `content_analyzer()`), and then it
    fits different VBPR algorithms using the ClayRS library depending on the number of epochs specified by the
    `-epo` cmd argument (invoking `recommender_system()`)

    The fit recommenders will be saved into `models/vbpr_clayrs`.

    """

    print("".center(80, '-'))

    if not os.path.isdir(os.path.join(DATA_DIR, "clayrs_output_ca")):
        content_analyzer_tradesy(os.path.join(PROCESSED_DIR, "item_map.csv"),
                                 features_matrix_path=os.path.join(PROCESSED_DIR, "features_matrix.npy"),
                                 output_contents_dir=os.path.join(DATA_DIR, "clayrs_output_ca"))
    else:
        print(f"Serialized contents are already present in {os.path.join(DATA_DIR, 'clayrs_output_ca')}, "
              f"content analyzer phase has been skipped")

    os.makedirs(os.path.join(MODEL_DIR, "vbpr_clayrs"), exist_ok=True)
    os.makedirs(os.path.join(YAML_DIR, "rs_report_comparison_exp"), exist_ok=True)

    print("".center(80, "-"))
    for epoch in ExperimentConfig.epochs:
        print(f"Considering number of epochs {epoch}")
        print("".center(80, '-'))
        cbrs = recsys_tradesy(train_set_path=os.path.join(PROCESSED_DIR, "train_set.csv"),
                              items_dir=os.path.join(DATA_DIR, "clayrs_output_ca"),
                              epoch=epoch)

        fname_cbrs = os.path.join(MODEL_DIR, "vbpr_clayrs", f"vbpr_clayrs_{epoch}.ml")

        with open(fname_cbrs, "wb") as file:
            pickle.dump(cbrs, file, protocol=pickle.HIGHEST_PROTOCOL)

        Report(output_dir=os.path.join(YAML_DIR, "rs_report_comparison_exp"),
               rs_report_filename=f"rs_report_{epoch}").yaml(recsys=cbrs)

        print(f"ClayRS model for {epoch} epochs saved into {fname_cbrs}!")
        print(f"Report of the RecSys phase for {epoch} epochs saved into "
              f"{os.path.join(YAML_DIR, 'rs_report_comparison_exp', f'rs_report_{epoch}')}.yml!")

        # if this is the last epoch we do not print the separator
        if epoch != ExperimentConfig.epochs[-1]:
            print("".center(80, '-'))


if __name__ == "__main__":
    main()
