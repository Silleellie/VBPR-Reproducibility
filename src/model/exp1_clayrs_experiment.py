"""
Module used by the `comparison` experiment.

Uses the NPY feature matrix built in the data preparation phase to fit the VBPR algorithm using the ClayRS framework.
"""

import os

import clayrs_can_see.content_analyzer as ca
from clayrs_can_see.utils import Report

from src import PROCESSED_DIR, DATA_DIR, MODEL_DIR, ExperimentConfig, YAML_DIR
from src.model import clayrs_recsys


def content_analyzer_tradesy(output_contents_dir: str):
    """
    Performs the Content Analyzer phase of the `comparison` experiment using the ClayRS framework.
    The representation that will be generated simply imports the original feature vectors and associates them
    to the corresponding item (thanks to the `FromNPY()` content technique).

    A .yml file containing the specified technique and its parameters is saved into the `reports/yaml_clayrs`
    directory.

    Args:
        output_contents_dir: path to the directory where the contents will be serialized

    """

    tradesy_item_map_path = os.path.join(PROCESSED_DIR, "item_map.csv")
    features_matrix_path = os.path.join(PROCESSED_DIR, "features_matrix.npy")

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
            ),
            id="imported_features"
        )
    )

    content_a = ca.ContentAnalyzer(items_ca, n_thread=ExperimentConfig.num_threads_ca)
    content_a.fit()

    Report(output_dir=YAML_DIR, ca_report_filename="exp1_ca_report").yaml(content_analyzer=content_a)

    print()
    print(f"Output of the Content Analyzer saved into {output_contents_dir}!")
    print(f"Report of the Content Analyzer phase saved into {os.path.join(YAML_DIR, 'exp1_ca_report.yml')}!")


def main():
    """
    Actual main function of the module.

    It first serializes the contents complexly represented (invoking `content_analyzer()`), and then it
    fits different VBPR algorithms using the ClayRS library depending on the number of epochs specified by the
    `-epo` cmd argument (invoking `recommender_system()`)

    The fit recommenders will be saved into `models/vbpr_clayrs`.

    """

    print("".center(80, '-'))

    output_contents_dir = os.path.join(DATA_DIR, "exp1_ca_output")

    if not os.path.isdir(output_contents_dir):
        content_analyzer_tradesy(output_contents_dir)
    else:
        print(f"Serialized contents are already present in {output_contents_dir}, "
              f"content analyzer phase has been skipped")

    print("".center(80, "-"))

    models_dir = os.path.join(MODEL_DIR, "exp1", "vbpr_clayrs")
    os.makedirs(models_dir, exist_ok=True)

    clayrs_recsys(contents_dir=output_contents_dir,
                  item_field="item_idx",
                  field_representation_list=["imported_features"],
                  exp_string="exp1",
                  models_dir=models_dir)


if __name__ == "__main__":
    main()
